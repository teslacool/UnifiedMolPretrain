from torch_geometric.data import InMemoryDataset
import shutil, os
import os.path as osp
import torch
import numpy as np
from tqdm import tqdm
from pretrain3d.data.pcqm4m import DGData
from pretrain3d.utils.graph import smiles2graphwithface
from rdkit import Chem
from copy import deepcopy
import glob
from rdkit.Chem.rdmolfiles import SDMolSupplier, MolFromMol2File
from openbabel import openbabel
import io
import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


class With3dPropPredDataset(InMemoryDataset):
    def __init__(
        self,
        name,
        root="dataset",
        transform=None,
        pre_transform=None,
        data_seed=42,
        basedir="/data/data/toxicity",
        valid_ratio=0.05,
        dataset_subset="LD50",
    ):
        self.name = name
        self.dirname = "with3dproppred_" + "_".join(name.split("-"))
        self.original_root = root
        self.root = osp.join(root, self.dirname)
        self.task_type = "regression"
        self.data_seed = data_seed
        self.basedir = basedir
        self.valid_ratio = valid_ratio
        self.dataset_subset = dataset_subset
        super().__init__(self.root, transform, pre_transform)
        self.data, self.slices, self.set2ids = torch.load(self.processed_paths[0])

    def get_idx_split(self,):
        train_idx = []
        valid_idx = []
        test_idx = []
        dataset_names = self.dataset_name_list
        for dataset in dataset_names:
            original_train = self.set2ids[f"{dataset}_training"]
            total_len = len(original_train)
            train_len = int(total_len * (1 - self.valid_ratio))
            _random_state = np.random.RandomState(seed=self.data_seed)

            original_train_perm = _random_state.permutation(original_train)
            train_idx.append(original_train_perm[:train_len])
            valid_idx.append(original_train_perm[train_len:])
            test_idx.extend(self.set2ids[f"{dataset}_prediction"])

        return {
            "train": torch.as_tensor(np.concatenate(train_idx, axis=0), dtype=torch.long),
            "valid": torch.as_tensor(np.concatenate(valid_idx, axis=0), dtype=torch.long),
            "test": torch.as_tensor(test_idx, dtype=torch.long),
        }

    @property
    def num_tasks(self):
        if "tox" in self.name:
            return 4
        else:
            return 1

    @property
    def raw_file_names(self):
        return ["data.npz"]

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        if osp.exists(self.processed_paths[0]):
            return
        else:
            assert os.path.exists(self.basedir)

    def process(self):
        if self.name == "tox":
            self.process_tox()
        else:
            raise NotImplementedError()

    @property
    def dataset_name_list(self):
        if self.name == "tox":
            if self.dataset_subset == "multitask":
                return ["IGC50", "LC50", "LC50DM", "LD50"]
            else:
                return [self.dataset_subset]
        else:
            raise ValueError(self.name)

    def process_tox(self):
        set2ids = dict()
        data_list = []
        for dataset_id, dataset in enumerate(["IGC50", "LC50", "LC50DM", "LD50"]):
            dataset_dir = osp.join(self.basedir, dataset)
            mol2dir = osp.join(dataset_dir, "schrodinger")
            tmpdir = osp.join("/tmp", dataset)
            os.makedirs(tmpdir, exist_ok=True)
            for sdfdir, tgtdir in zip(
                ["training", "prediction"], ["training_target", "prediction_target"]
            ):
                set2ids[f"{dataset}_{sdfdir}"] = []
                bad_cases = 0
                sdffns = glob.glob(os.path.join(dataset_dir, sdfdir, "*.sdf"))
                for sdffn in tqdm(sdffns):
                    sdfid = os.path.splitext(os.path.basename(sdffn))[0]
                    mol2fn = osp.join(mol2dir, f"{sdfid}.mol2")
                    if not osp.exists(mol2fn):
                        print(f"{mol2fn} does not exist")
                        bad_cases += 1
                        continue
                    mol = MolFromMol2File(mol2fn)
                    if mol is None:
                        conv = openbabel.OBConversion()
                        conv.SetInAndOutFormats("mol2", "sdf")
                        sdffn = osp.join(tmpdir, f"{sdfid}.sdf")
                        conv.OpenInAndOutFiles(mol2fn, sdffn)
                        conv.Convert()
                        conv.CloseOutFile()

                        if mol is None:
                            mol = next(SDMolSupplier(sdffn, sanitize=False, strictParsing=False))
                            mol.UpdatePropertyCache(strict=False)
                            if mol is None:
                                bad_cases += 1
                                continue
                    # try:
                    graph = smiles2graphwithface(mol)
                    # except:
                    #     bad_cases += 1
                    #     continue
                    data = DGData()

                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]

                    data.__num_nodes__ = int(graph["num_nodes"])
                    data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)

                    target_fn = osp.join(dataset_dir, tgtdir, f"{sdfid}.exp")
                    with io.open(target_fn, "r", encoding="utf8", newline="\n") as f:
                        label = float(f.readline().strip())
                    data.y = (
                        torch.as_tensor([np.nan, np.nan, np.nan, np.nan])
                        .view(1, -1)
                        .to(torch.float32)
                    )
                    data.y[0, dataset_id] = label
                    data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
                    data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
                    data.nf_node = torch.from_numpy(graph["nf_node"]).to(torch.int64)
                    data.nf_ring = torch.from_numpy(graph["nf_ring"]).to(torch.int64)
                    data.num_rings = int(graph["num_rings"])
                    data.n_edges = int(graph["n_edges"])
                    data.n_nodes = int(graph["n_nodes"])
                    data.n_nfs = int(graph["n_nfs"])
                    data.rdmol = deepcopy(mol)
                    data.sdf_id = sdfid
                    data.pos = torch.from_numpy(mol.GetConformer(0).GetPositions()).to(torch.float)
                    set2ids[f"{dataset}_{sdfdir}"].append(len(data_list))
                    data_list.append(data)

                print(f"{dataset} {sdfdir} bad/all {bad_cases}/{len(sdffns)}")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices, set2ids), self.processed_paths[0])


if __name__ == "__main__":
    dataset = With3dPropPredDataset("tox")
    result = dataset.get_idx_split()
    print(result)

