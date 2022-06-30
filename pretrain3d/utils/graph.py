from rdkit import Chem
from pretrain3d.utils.features import atom_to_feature_vector, bond_to_feature_vector
import numpy as np


def getface(mol):
    assert isinstance(mol, Chem.Mol)
    bond2id = dict()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond2id[(i, j)] = len(bond2id)
        bond2id[(j, i)] = len(bond2id)

    num_edge = len(bond2id)
    left = [0] * num_edge
    ssr = Chem.GetSymmSSSR(mol)
    face = [[]]
    for ring in ssr:
        ring = list(ring)

        bond_list = []
        for i, atom in enumerate(ring):
            bond_list.append((ring[i - 1], atom))

        exist = False
        if any([left[bond2id[bond]] != 0 for bond in bond_list]):
            exist = True
        if exist:
            ring = list(reversed(ring))
        face.append(ring)
        for i, atom in enumerate(ring):
            bond = (ring[i - 1], atom)
            if left[bond2id[bond]] != 0:
                bond = (atom, ring[i - 1])
            bondid = bond2id[bond]
            if left[bondid] == 0:
                left[bondid] = len(face) - 1

    return face, left, bond2id


def smiles2graphwithface(mol):

    # mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features_list, dtype=np.int64)

        faces, left, _ = getface(mol)
        num_faces = len(faces)
        face_mask = [False] * num_faces
        face_index = [[-1, -1]] * len(edges_list)
        face_mask[0] = True
        for i in range(len(edges_list)):
            inface = left[i ^ 1]
            outface = left[i]
            face_index[i] = [inface, outface]

        nf_node = []
        nf_ring = []
        for i, face in enumerate(faces):
            face = list(set(face))
            nf_node.extend(face)
            nf_ring.extend([i] * len(face))

        face_mask = np.array(face_mask, dtype=np.bool)
        face_index = np.array(face_index, dtype=np.int64).T
        n_nfs = len(nf_node)
        nf_node = np.array(nf_node, dtype=np.int64).reshape(1, -1)
        nf_ring = np.array(nf_ring, dtype=np.int64).reshape(1, -1)

    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)
        face_mask = np.empty((0), dtype=np.bool)
        face_index = np.empty((2, 0), dtype=np.int64)
        num_faces = 0
        n_nfs = 0
        nf_node = np.empty((1, 0), dtype=np.int64)
        nf_ring = np.empty((1, 0), dtype=np.int64)

    n_src = list()
    n_tgt = list()
    for atom in mol.GetAtoms():
        n_ids = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(n_ids) > 1:
            n_src.append(atom.GetIdx())
            n_tgt.append(n_ids[:6])
    nums_neigh = len(n_src)
    nei_src_index = np.array(n_src, dtype=np.int64).reshape(1, -1)
    nei_tgt_index = np.zeros((6, nums_neigh), dtype=np.int64)
    nei_tgt_mask = np.ones((6, nums_neigh), dtype=np.bool)

    for i, n_ids in enumerate(n_tgt):
        nei_tgt_index[: len(n_ids), i] = n_ids
        nei_tgt_mask[: len(n_ids), i] = False

    graph = dict()
    graph["edge_index"] = edge_index
    graph["edge_feat"] = edge_attr
    graph["node_feat"] = x
    graph["num_nodes"] = len(x)

    # we do not use the keyword "face", since "face" is already used by torch_geometric.
    graph["ring_mask"] = face_mask
    graph["ring_index"] = face_index
    graph["num_rings"] = num_faces
    graph["n_edges"] = len(edge_attr)
    graph["n_nodes"] = len(x)

    graph["n_nfs"] = n_nfs
    graph["nf_node"] = nf_node
    graph["nf_ring"] = nf_ring

    graph["nei_src_index"] = nei_src_index
    graph["nei_tgt_index"] = nei_tgt_index
    graph["nei_tgt_mask"] = nei_tgt_mask

    return graph


if __name__ == "__main__":
    smiles_string = r"[N+]12CCC(CC1)C(OC(=O)C(O)(c1ccccc1)c1ccccc1)C2.[Br-]"
    mol = Chem.MolFromSmiles(smiles_string)

    faces, left, bond2id = getface(mol)
    for i, face in enumerate(faces):
        print(i, *face)
    for bond, idx in bond2id.items():
        print(bond, left[idx])
