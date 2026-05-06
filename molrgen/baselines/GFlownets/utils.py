import os.path
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

rdBase.DisableLog("rdApp.error")


atomic_numbers = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
}


def get_fragments(
    path: str = "data/fragdb/example_blocks.json",
) -> List[Tuple[str, List[int]]]:
    assert os.path.exists(path)
    df = pd.read_json(path)
    return [(smi, r) for smi, r in zip(df["block_smi"], df["block_r"])]


def compute_isometry(mol: Chem.Mol) -> np.ndarray:
    ":return [num_atoms] isometric group"
    isom_groups = list(Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
    return np.asarray(isom_groups, dtype=np.int32)


def break_on_bonds(
    mol: Chem.Mol, jun_bonds: np.ndarray, frags_generic: bool
) -> Tuple[
    np.ndarray,
    List[str],
    List[List[str]],
    List[np.ndarray],
    List[List[Tuple[int, int]]],
]:
    """Breaks molecule into fragments
    :param mol: molecule
    :param junction_bonds: junction_bonds [n_junction_bonds,4] bond = [frag0, frag1, atm0, atm1]
    :return:
    """
    # fixme break ties
    # todo bondtypes (?)
    # todo assert jbonds num atoms
    n_junct = jun_bonds.shape[0]
    # break bonds of the molecule; get fragments
    Chem.Kekulize(mol, clearAromaticFlags=False)
    emol = Chem.EditableMol(mol)
    [
        emol.RemoveBond(int(jun_bonds[i, 0]), int(jun_bonds[i, 1]))
        for i in range(n_junct)
    ]
    frags = list(Chem.GetMolFrags(emol.GetMol(), asMols=True, sanitizeFrags=True))

    if frags_generic:
        frags = [MurckoScaffold.MakeScaffoldGeneric(frag) for frag in frags]
    assert len(frags) == n_junct + 1, (
        "internal error: molecule did not break into the right number of fragments"
    )
    # get a frag_name for each of the groups
    frag_names = [Chem.MolToSmiles(f) for f in frags]
    # reorder atom numbers to follow exactly the SMILES string we just took
    frag_canonord = [
        frags[i].GetPropsAsDict(includePrivate=True, includeComputed=True)[
            "_smilesAtomOutputOrder"
        ]
        for i in range(n_junct + 1)
    ]
    mol_canonord = list(np.asarray(frag) for frag in Chem.GetMolFrags(emol.GetMol()))
    mol_canonord = [
        mol_canonord[i][np.asarray(frag_canonord[i])] for i in range(n_junct + 1)
    ]
    mol_canonord = np.concatenate(mol_canonord, 0)
    frags = [Chem.RenumberAtoms(frags[i], frag_canonord[i]) for i in range(n_junct + 1)]
    mol = Chem.RenumberAtoms(mol, [int(idx) for idx in mol_canonord])
    frag_startidx = np.concatenate(
        [[0], np.cumsum([frag.GetNumAtoms() for frag in frags])], 0
    )[:-1]
    # get fragment elements frag_elem and frag_coord
    frag_atoms = [frags[i].GetAtoms() for i in range(n_junct + 1)]
    frag_elem = []
    frag_coord = []
    for i in range(n_junct + 1):
        frag_natm = len(frag_atoms[i])
        frag_elem.append([frag_atoms[i][j].GetSymbol() for j in range(frag_natm)])
        if frags[i].GetNumConformers() > 0:
            frag_coord.append(
                np.asarray(
                    [
                        frags[i].GetConformer(0).GetAtomPosition(j)
                        for j in range(frag_natm)
                    ]
                )
            )
        else:
            frag_coord.append(None)
    # find junction/fragment bonds for each of the groups
    bonds = np.asarray(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in list(mol.GetBonds())]
    ).reshape([-1, 2])
    bond_frags = np.searchsorted(frag_startidx, bonds, side="right") - 1
    bond_atms = bonds - frag_startidx[bond_frags]
    bonds = np.concatenate([bond_frags, bond_atms], 1)
    is_junction = [bond_frags[:, 0] != bond_frags[:, 1]][0]
    jun_bonds = bonds[is_junction]
    assert jun_bonds.shape[0] == n_junct, "internal error in fragmentation"
    frag_bonds = [[] for _ in range(n_junct + 1)]  # type: ignore
    for bond in bonds[np.logical_not(is_junction)]:
        frag_bonds[bond[0]].append(bond[2:])
    return jun_bonds, frag_names, frag_elem, frag_coord, frag_bonds


def find_rota_murcko_bonds(
    mol: Chem.Mol, sanitize: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds rotatable bonds (first) and Murcko bonds (second)
    :param mol:
    :return:
    """
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            raise ValueError("error in sanitization") from e

    # r_groups decomposition
    rotatable = Chem.MolFromSmarts(
        "[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]"
    )
    rota_bonds = mol.GetSubstructMatches(rotatable)
    r_bonds = [((x, y), (0, 0)) for x, y in rota_bonds]
    r_groups = BRICS.BreakBRICSBonds(mol, r_bonds)
    r_atmids = list(Chem.GetMolFrags(r_groups))
    r_groups = list(Chem.GetMolFrags(r_groups, asMols=True))

    # Murcko decomposition
    murcko_bonds = []  # murcko bonds
    for rgroup_num in range(len(r_groups)):
        m_core = MurckoScaffold.GetScaffoldForMol(r_groups[rgroup_num])
        if m_core.GetNumAtoms() > 0:
            # get all bonds in this fragment
            rg_bonds = [
                [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
                for bond in r_groups[rgroup_num].GetBonds()
            ]
            # print ([bond.GetBondType() for bond in r_groups[rgroup_num].GetBonds()])
            rg_bonds = np.asarray(rg_bonds)
            # remove bonds that lead to dummy atoms added by BreakBRICSBonds
            dummy_atoms = np.asarray(r_atmids[rgroup_num]) >= mol.GetNumAtoms()
            dummy_bonds = dummy_atoms[rg_bonds.reshape([-1])].reshape([-1, 2])
            dummy_bonds = np.logical_or(dummy_bonds[:, 0], dummy_bonds[:, 1])
            rg_bonds = rg_bonds[np.logical_not(dummy_bonds)]
            # filter out only the ones that connect core with something outside of the core
            mcore_atmid = np.asarray(r_groups[rgroup_num].GetSubstructMatch(m_core))
            m_bonds = np.reshape(np.in1d(rg_bonds, mcore_atmid), [-1, 2])
            m_bonds = rg_bonds[np.logical_xor(m_bonds[:, 0], m_bonds[:, 1])]
            # return bond-numbering to the one for a whole molecule
            m_bonds = np.asarray(r_atmids[rgroup_num])[np.reshape(m_bonds, [-1])]
            m_bonds = np.reshape(m_bonds, [-1, 2])
            [murcko_bonds.append(m_bond) for m_bond in m_bonds]  # type: ignore

    rota_bonds = np.reshape(np.asarray(rota_bonds, dtype=np.int64), [-1, 2])
    murcko_bonds = np.reshape(np.asarray(murcko_bonds, dtype=np.int64), [-1, 2])
    return rota_bonds, murcko_bonds


def fragment_molecule(
    mol: Chem.Mol, frags_generic: bool, decomposition: str
) -> Tuple[
    List[List[str]],
    List[np.ndarray],
    List[List[Tuple[int, int]]],
    np.ndarray,
    List[str],
]:
    """Fragments a whole molecule and returns fragment information.
    :param mol: RDKit molecule to fragment
    :param frags_generic: whether to make fragments generic using Murcko scaffold
    :param decomposition: type of decomposition to use ("rota_murcko")
    :return: tuple of (frag_elem, frag_coord, frag_bonds, jun_bonds, frag_names)
    """
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        raise ValueError("error in sanitization") from e
    try:
        if decomposition == "rota_murcko":
            rota_bonds, murcko_bonds = find_rota_murcko_bonds(mol)
            jun_bonds = np.concatenate([rota_bonds, murcko_bonds], axis=0)
        else:
            raise NotImplementedError(
                f"decomposition method '{decomposition}' not implemented"
            )
        jun_bonds, frag_names, frag_elem, frag_coord, frag_bonds = break_on_bonds(
            mol, jun_bonds, frags_generic
        )
    except Exception as e:
        raise ValueError("error in fragmenting") from e
    return frag_elem, frag_coord, frag_bonds, jun_bonds, frag_names
