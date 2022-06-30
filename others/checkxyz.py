from openbabel import openbabel
import pandas as pd
import os
import torch
from tqdm import tqdm
from rdkit.Chem.rdmolfiles import SDMolSupplier
from rdkit import Chem

rawdir = "examples/lsc/pcqm4m-v2/dataset/pcqm4m-v2/raw"
data_df = pd.read_csv(os.path.join(rawdir, "data.csv.gz"))
smiles_list = data_df["smiles"]

datadir = "examples/lsc/pcqm4m-v2/dataset/pcqm4m-v2"
split_dict = torch.load(os.path.join(datadir, "split_dict.pt"))
train_idxs = split_dict["train"].tolist()

xyzdir = "/data/data/pcqm4m-v2_xyz"
bad_cases = 0
for i in tqdm(train_idxs):
    prefix = i // 10000
    prefix = "{0:04d}0000_{0:04d}9999".format(prefix)
    xyzfn = os.path.join(xyzdir, prefix, f"{i}")
    conv = openbabel.OBConversion()
    conv.OpenInAndOutFiles(f"{xyzfn}.xyz", f"{xyzfn}.sdf")
    conv.SetInAndOutFormats("xyz", "sdf")
    conv.Convert()
    conv.CloseOutFile()
    suppl = SDMolSupplier(f"{xyzfn}.sdf")
    mol = next(suppl)
    try:
        assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles_list[i]))
    except:
        # print(i)
        # print(Chem.MolToSmiles(mol))
        # print(smiles_list[i])
        # print(Chem.MolToSmiles(Chem.MolFromSmiles(smiles_list[i])))
        bad_cases += 1
print(bad_cases, len(train_idxs))
