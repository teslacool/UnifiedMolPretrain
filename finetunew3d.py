import os
import argparse
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
import io
from pretrain3d.data.With3dPropPredDataset.dataset import With3dPropPredDataset
from pretrain3d.data.Qm9Dataset.dataset import Qm9Dataset
from pretrain3d.model.gnn import GNNet
from torch.optim.lr_scheduler import LambdaLR
from pretrain3d.utils.misc import WarmCosine, PreprocessBatch, WarmLinear
from pretrain3d.utils.dist import init_distributed_mode
import json
from torch.utils.data import DistributedSampler
from pretrain3d.utils.evaluate import Evaluatorwith3d, EvaluatorQm9
import torch.nn.functional as F
import pickle


def train(model, device, train_loader, optimizer, scheduler, args, preprocessor, task_type):
    model.train()
    loss_accum = 0
    pbar = tqdm(train_loader, desc="Iteration", disable=args.disable_tqdm)
    for step, batch in enumerate(pbar):
        batch = batch.to(device)
        preprocessor.process(batch)
        optimizer.zero_grad()
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred_attrs, attr_mask_index, pos_predictions, pos_mask_idx = model(batch, mode="raw")
            is_labeled = batch.y == batch.y

            if "classification" in task_type:
                loss = F.binary_cross_entropy_with_logits(
                    pred_attrs.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled]
                )
            else:
                if "qm9" in args.dataset:
                    loss = F.l1_loss(
                        pred_attrs.to(torch.float32)[is_labeled],
                        batch.y.to(torch.float32)[is_labeled],
                    )
                else:
                    loss = F.mse_loss(
                        pred_attrs.to(torch.float32)[is_labeled],
                        batch.y.to(torch.float32)[is_labeled],
                    )
            loss.backward()
            if args.grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            scheduler.step()

            loss_accum += loss.detach().item()
            if step % args.log_interval == 0:
                pbar.set_description(
                    "Iteration loss: {:6.4f} lr: {:.5e}".format(
                        loss_accum / (step + 1), scheduler.get_last_lr()[0]
                    )
                )

    return {"loss": loss_accum / (step + 1)}


def evaluate(model, device, loader, args, preprocessor, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for step, batch in enumerate(
            tqdm(loader, desc="Valid Iteration", disable=args.disable_tqdm)
        ):
            batch = batch.to(device)
            preprocessor.process(batch)
            pred_attrs, attr_mask_index, pos_predictions, pos_mask_idx = model(batch, mode="raw")
            y_true.append(batch.y.view(pred_attrs.shape).detach().cpu())
            y_pred.append(pred_attrs.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict), input_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--global-reducer", type=str, default="sum")
    parser.add_argument("--node-reducer", type=str, default="sum")
    parser.add_argument("--graph-pooling", type=str, default="sum")
    parser.add_argument("--dropedge-rate", type=float, default=0.1)
    parser.add_argument("--dropnode-rate", type=float, default=0.1)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--latent-size", type=int, default=256)
    parser.add_argument("--mlp-hidden-size", type=int, default=1024)
    parser.add_argument("--mlp-layers", type=int, default=2)
    parser.add_argument("--use-layer-norm", action="store_true", default=False)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=str, default="")

    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--encoder-dropout", type=float, default=0.0)
    parser.add_argument("--pooler-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--layernorm-before", action="store_true", default=False)
    parser.add_argument("--use-bn", action="store_true", default=False)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--period", type=float, default=10)
    parser.add_argument("--enable-tb", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use-face", action="store_true", default=False)
    parser.add_argument("--global-attn", action="store_true", default=False)
    parser.add_argument("--node-attn", action="store_true", default=False)
    parser.add_argument("--face-attn", action="store_true", default=False)
    parser.add_argument("--grad-norm", type=float, default=None)
    parser.add_argument("--random-rotation", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--pred-pos-residual", action="store_true", default=False)
    parser.add_argument("--raw-with-pos", action="store_false", default=True)
    parser.add_argument("--finetune-from", type=str, default="checkpoints/checkpoint_94.pt")
    parser.add_argument("--dataset", type=str, default="tox")
    parser.add_argument("--dataset-subset", type=str, default="LD50")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--ap-hid-size", type=int, default=512)
    parser.add_argument("--ap-mlp-layers", type=int, default=2)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--gradmultiply", type=float, default=0.1)

    args = parser.parse_args()
    init_distributed_mode(args)
    print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    if args.dataset.startswith("qm9"):
        dataset = Qm9Dataset(data_seed=args.data_seed)
    else:
        dataset = With3dPropPredDataset(
            args.dataset,
            data_seed=args.data_seed,
            dataset_subset=args.dataset_subset,
            valid_ratio=args.valid_ratio,
        )
    split_idx = dataset.get_idx_split()
    if args.dataset.startswith("qm9"):
        evaluator = EvaluatorQm9(dataset)
    else:
        evaluator = Evaluatorwith3d(args.dataset)

    dataset_train = dataset[split_idx["train"]]
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )
    train_loader = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    shared_params = dict(
        mlp_hidden_size=args.mlp_hidden_size,
        mlp_layers=args.mlp_layers,
        latent_size=args.latent_size,
        use_layer_norm=args.use_layer_norm,
        num_message_passing_steps=args.num_layers,
        global_reducer=args.global_reducer,
        node_reducer=args.node_reducer,
        face_reducer=args.node_reducer,
        dropedge_rate=args.dropedge_rate,
        dropnode_rate=args.dropnode_rate,
        use_face=args.use_face,
        dropout=args.dropout,
        graph_pooling=args.graph_pooling,
        layernorm_before=args.layernorm_before,
        encoder_dropout=args.encoder_dropout,
        pooler_dropout=args.pooler_dropout,
        use_bn=args.use_bn,
        global_attn=args.global_attn,
        node_attn=args.node_attn,
        face_attn=args.face_attn,
        pred_pos_residual=args.pred_pos_residual,
        raw_with_pos=args.raw_with_pos,
        attr_predict=True,
        num_tasks=dataset.num_tasks,
        ap_hid_size=args.ap_hid_size,
        ap_mlp_layers=args.ap_mlp_layers,
        gradmultiply=args.gradmultiply,
    )
    model = GNNet(**shared_params).to(device)
    print(model)
    if args.finetune_from is not None:
        assert os.path.exists(args.finetune_from)
        print(f"Load pre-trained model from {args.finetune_from}")
        checkpoint = torch.load(args.finetune_from, map_location=torch.device("cpu"))[
            "model_state_dict"
        ]
        model.load_state_dict(checkpoint)

    preprocessor = PreprocessBatch(True if args.raw_with_pos else False, args.random_rotation)
    args.disable_tqdm = False

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        args.checkpoint_dir = "" if args.rank != 0 else args.checkpoint_dir
        args.enable_tb = False if args.rank != 0 else args.enable_tb
        args.disable_tqdm = args.rank != 0
    num_params = sum(p.numel() for p in model_without_ddp.parameters())
    print(f"#Params: {num_params}")

    optimizer = optim.AdamW(
        model_without_ddp.parameters(),
        lr=args.lr,
        betas=(0.9, args.beta2),
        weight_decay=args.weight_decay,
    )
    lrscheduler = WarmLinear(
        tmax=len(train_loader) * args.epochs, warmup=len(train_loader) * args.epochs * 0.06
    )
    scheduler = LambdaLR(optimizer, lambda x: lrscheduler.step(x))

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.checkpoint_dir and args.enable_tb:
        tb_writer = SummaryWriter(args.checkpoint_dir)

    # valid_curve = []
    # test_curve = []
    # train_curve = []

    for epoch in range(1, args.epochs + 1):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train_loss_dict = train(
            model, device, train_loader, optimizer, scheduler, args, preprocessor, dataset.task_type
        )

        print("Evaluating...")
        train_perf, _ = evaluate(model, device, train_loader, args, preprocessor, evaluator)
        valid_perf, valid_output = evaluate(
            model, device, valid_loader, args, preprocessor, evaluator
        )
        test_perf, test_output = evaluate(model, device, test_loader, args, preprocessor, evaluator)
        if args.checkpoint_dir:
            print(f"Setting {os.path.basename(os.path.normpath(args.checkpoint_dir))}...")

        print("Train", train_perf, "Validation", valid_perf, "Test", test_perf)
        # train_curve.append(train_perf[dataset.eval_metric])
        # valid_curve.append(valid_perf[dataset.eval_metric])
        # test_curve.append(test_perf[dataset.eval_metric])

        if args.checkpoint_dir:
            logs = {
                "epoch": epoch,
                "Train": train_perf,
                "Validation": valid_perf,
                "Test": test_perf,
            }
            with io.open(
                os.path.join(args.checkpoint_dir, "log.txt"), "a", encoding="utf8", newline="\n"
            ) as tgt:
                print(json.dumps(logs), file=tgt)
            with open(os.path.join(args.checkpoint_dir, f"output_{epoch}.pkl"), "wb") as f:
                pickle.dump((valid_output, test_output), f)

            if args.enable_tb:
                for k, v in train_loss_dict.items():
                    tb_writer.add_scalar(f"train/{k}", v, epoch)
                for dataset_name, output_dict in train_perf.items():
                    for metric, value in output_dict.items():
                        tb_writer.add_scalar(f"eval/train_{dataset_name}_{metric}", value, epoch)
                for dataset_name, output_dict in valid_perf.items():
                    for metric, value in output_dict.items():
                        tb_writer.add_scalar(f"eval/valid_{dataset_name}_{metric}", value, epoch)
                for dataset_name, output_dict in test_perf.items():
                    for metric, value in output_dict.items():
                        tb_writer.add_scalar(f"eval/test_{dataset_name}_{metric}", value, epoch)

                # tb_writer.add_scalar(
                #     f"eval/train_{dataset.eval_metric}", train_perf[dataset.eval_metric], epoch
                # )
                # tb_writer.add_scalar(
                #     f"eval/valid_{dataset.eval_metric}", valid_perf[dataset.eval_metric], epoch
                # )
                # tb_writer.add_scalar(
                #     f"eval/test_{dataset.eval_metric}", test_perf[dataset.eval_metric], epoch
                # )

    # if "classification" in dataset.task_type:
    #     best_val_epoch = np.argmax(np.array(valid_curve))
    # else:
    #     best_val_epoch = np.argmin(np.array(valid_curve))

    print("Finished training!")
    # print("Best validation score: {}".format(valid_curve[best_val_epoch]))
    # print("Test score: {}".format(test_curve[best_val_epoch]))


if __name__ == "__main__":
    main()

