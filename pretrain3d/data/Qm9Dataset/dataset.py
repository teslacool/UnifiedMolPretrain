from torch_geometric.data import InMemoryDataset
import os
import os.path as osp
import torch
from pretrain3d.data.pcqm4m import DGData
from pretrain3d.utils.graph import smiles2graphwithface
from rdkit import Chem
from copy import deepcopy
from scipy.constants import physical_constants
import numpy as np
import pandas as pd
from tqdm import tqdm

hartree2eV = physical_constants["hartree-electron volt relationship"][0]


class Qm9Dataset(InMemoryDataset):
    def __init__(
        self, name="qm9", root="dataset", transform=None, pre_transform=None, data_seed=42,
    ):
        self.name = name
        self.dirname = f"{name}_v1"
        self.original_root = root
        self.root = osp.join(root, self.dirname)
        self.task_type = "regression"

        self.unit_conversion = {
            # "A": 1.0,
            # "B": 1.0,
            # "C": 1.0,
            "mu": 1.0,
            "alpha": 1.0,
            "homo": hartree2eV,
            "lumo": hartree2eV,
            "gap": hartree2eV,
            "r2": 1.0,
            "zpve": hartree2eV,
            "u0": hartree2eV,
            "u298": hartree2eV,
            "h298": hartree2eV,
            "g298": hartree2eV,
            "cv": 1.0,
            # "u0_atom": hartree2eV,
            # "u298_atom": hartree2eV,
            # "h298_atom": hartree2eV,
            # "g298_atom": hartree2eV,
        }
        self.unit_conversion_values = [
            1.0,
            1.0,
            hartree2eV,
            hartree2eV,
            hartree2eV,
            1.0,
            hartree2eV,
            hartree2eV,
            hartree2eV,
            hartree2eV,
            hartree2eV,
            1.0,
        ]
        self.target_tasks = [
            "mu",
            "alpha",
            "homo",
            "lumo",
            "gap",
            "r2",
            "zpve",
            "u0",
            "u298",
            "h298",
            "g298",
            "cv",
        ]
        for target_task in self.target_tasks:
            assert target_task in self.unit_conversion

        super().__init__(self.root, transform, pre_transform)

        self.data, self.slices, self.target_mean, self.target_std = torch.load(
            self.processed_paths[0]
        )
        self.data_seed = data_seed
        self.eV2meV = np.array(
            [1.0 if self.unit_conversion[task] == 1.0 else 1000.0 for task in self.target_tasks],
            dtype=np.float,
        ).reshape(1, -1)

    def get_idx_split(self):
        all_idx = np.arange(self.len())
        _random_state = np.random.RandomState(seed=self.data_seed)
        all_idx = _random_state.permutation(all_idx)
        model_idx = all_idx[:100000]
        test_idx = all_idx[len(model_idx) : len(model_idx) + int(0.1 * len(all_idx))]
        val_idx = all_idx[len(model_idx) + len(test_idx) :]
        train_idx = model_idx[:50000]
        return {
            "train": torch.as_tensor(train_idx, dtype=torch.long),
            "valid": torch.as_tensor(val_idx, dtype=torch.long),
            "test": torch.as_tensor(test_idx, dtype=torch.long),
        }

    @property
    def num_tasks(self):
        return len(self.target_tasks)

    @property
    def raw_file_names(self):
        return ["data.npz"]

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        if not os.path.exists(self.processed_paths[0]):
            filepath = os.path.dirname(__file__)
            assert os.path.exists(os.path.join(filepath, "qm9_eV.npz"))
            assert os.path.exists(os.path.join(filepath, "qm9.csv"))

    def process(self):
        filepath = os.path.dirname(__file__)
        data_qm9 = dict(np.load(os.path.join(filepath, "qm9_eV.npz"), allow_pickle=True))
        coordinates = torch.tensor(data_qm9["R"], dtype=torch.float)
        molecules_df = pd.read_csv(os.path.join(filepath, "qm9.csv"))

        total_atoms = 0
        print("Converting SMILES strings into graphs...")
        data_list = []

        for mol_idx, n_atoms in tqdm(enumerate(data_qm9["N"]), total=data_qm9["N"].shape[0]):
            mol = Chem.MolFromSmiles(molecules_df["smiles"][data_qm9["id"][mol_idx]])
            mol = Chem.AddHs(mol)

            rdkit_atomicnum = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            numpy_atomicnum = data_qm9["Z"][total_atoms : total_atoms + n_atoms].tolist()
            assert all(
                [rdkit_atomicnum[i] == numpy_atomicnum[i]] for i in range(len(rdkit_atomicnum))
            )

            data = DGData()
            graph = smiles2graphwithface(mol)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)

            target = torch.tensor(
                molecules_df.iloc[data_qm9["id"][mol_idx]][5:-4], dtype=torch.float
            )
            data.y = target.view(1, -1) * torch.tensor(self.unit_conversion_values)[None, :]

            data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
            data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
            data.nf_node = torch.from_numpy(graph["nf_node"]).to(torch.int64)
            data.nf_ring = torch.from_numpy(graph["nf_ring"]).to(torch.int64)
            data.num_rings = int(graph["num_rings"])
            data.n_edges = int(graph["n_edges"])
            data.n_nodes = int(graph["n_nodes"])
            data.n_nfs = int(graph["n_nfs"])
            data.rdmol = deepcopy(mol)

            pos = coordinates[total_atoms : total_atoms + n_atoms]
            total_atoms += n_atoms
            data.pos = pos
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print("Saving.....")
        y = data.y
        target_mean = torch.mean(y, dim=0, keepdim=True)
        target_std = torch.std(y, dim=0, keepdim=True)
        data.y = (y - target_mean) / (target_std + 1e-6)
        torch.save((data, slices, target_mean, target_std), self.processed_paths[0])


if __name__ == "__main__":
    dataset = Qm9Dataset()
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

