from torch_geometric.data import InMemoryDataset
import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from pretrain3d.utils.url import decide_download, download_url, extract_zip
from tqdm import tqdm
from pretrain3d.data.pcqm4m import DGData
from pretrain3d.utils.graph import smiles2graphwithface
from rdkit import Chem
from copy import deepcopy


class OgbGraphPropPredDataset(InMemoryDataset):
    def __init__(self, name, root="dataset", transform=None, pre_transform=None):
        self.name = name
        self.dirname = "ogbgraphproppred_" + "_".join(name.split("-"))
        self.original_root = root
        self.root = osp.join(root, self.dirname)

        master = pd.read_csv(osp.join(osp.dirname(__file__), "master.csv"), index_col=0)
        if not self.name in master:
            error_mssg = "Invalid dataset name {}.\n".format(self.name)
            error_mssg += "Available datasets are as follows:\n"
            error_mssg += "\n".join(master.keys())
            raise ValueError(error_mssg)
        self.meta_info = master[self.name]
        if osp.isdir(self.root) and (
            not osp.exists(
                osp.join(self.root, "RELEASE_v" + str(self.meta_info["version"]) + ".txt")
            )
        ):
            print(self.name + " has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.root)

        self.download_name = self.meta_info["download_name"]
        self.num_tasks = int(self.meta_info["num tasks"])
        self.eval_metric = self.meta_info["eval metric"]
        self.task_type = self.meta_info["task type"]
        self.__num_classes__ = int(self.meta_info["num classes"])
        self.binary = self.meta_info["binary"] == "True"

        super().__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info["split"]

        path = osp.join(self.root, "split", split_type)
        if os.path.isfile(os.path.join(path, "split_dict.pt")):
            return torch.load(os.path.join(path, "split_dict.pt"))

        train_idx = pd.read_csv(
            osp.join(path, "train.csv.gz"), compression="gzip", header=None
        ).values.T[0]
        valid_idx = pd.read_csv(
            osp.join(path, "valid.csv.gz"), compression="gzip", header=None
        ).values.T[0]
        test_idx = pd.read_csv(
            osp.join(path, "test.csv.gz"), compression="gzip", header=None
        ).values.T[0]

        return {
            "train": torch.tensor(train_idx, dtype=torch.long),
            "valid": torch.tensor(valid_idx, dtype=torch.long),
            "test": torch.tensor(test_idx, dtype=torch.long),
        }

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        if self.binary:
            return ["data.npz"]
        else:
            file_names = ["edge"]
            if self.meta_info["has_node_attr"] == "True":
                file_names.append("node-feat")
            if self.meta_info["has_edge_attr"] == "True":
                file_names.append("edge-feat")
            return [file_name + ".csv.gz" for file_name in file_names]

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        url = self.meta_info["url"]
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)

        else:
            print("Stop downloading.")
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        smiles_list = pd.read_csv(osp.join(self.root, "mapping", "mol.csv.gz"), compression="gzip")[
            "smiles"
        ].values
        labels = pd.read_csv(
            osp.join(self.raw_dir, "graph-label.csv.gz"), compression="gzip", header=None
        ).values
        has_nan = np.isnan(labels).any()

        print("Converting SMILES strings into graphs...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = DGData()
            smiles = smiles_list[i]
            mol = Chem.MolFromSmiles(smiles)
            graph = smiles2graphwithface(mol)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            if "classification" in self.task_type and (not has_nan):
                data.y = torch.from_numpy(labels[i]).view(1, -1).to(torch.long)
            else:
                data.y = torch.from_numpy(labels[i]).view(1, -1).to(torch.float32)

            data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
            data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
            data.nf_node = torch.from_numpy(graph["nf_node"]).to(torch.int64)
            data.nf_ring = torch.from_numpy(graph["nf_ring"]).to(torch.int64)
            data.num_rings = int(graph["num_rings"])
            data.n_edges = int(graph["n_edges"])
            data.n_nodes = int(graph["n_nodes"])
            data.n_nfs = int(graph["n_nfs"])
            data.rdmol = deepcopy(mol)
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    dataset = OgbGraphPropPredDataset("ogbg-molpcba")
    print(dataset.num_classes)
    split_index = dataset.get_idx_split()
    print(dataset[0])
    # print(pyg_dataset[0].node_is_attributed)
    print([dataset[i].x[1] for i in range(100)])
    # print(pyg_dataset[0].y)
    # print(pyg_dataset[0].edge_index)
    print(dataset[split_index["train"]])
    print(dataset[split_index["valid"]])
    print(dataset[split_index["test"]])

