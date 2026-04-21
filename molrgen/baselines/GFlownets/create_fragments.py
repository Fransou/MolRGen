# sys.path.append("../../")
import json
import os
import pickle
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rdkit import Chem, rdBase
from tqdm import tqdm

from molrgen.baselines.GFlownets.utils import (
    atomic_numbers,
    compute_isometry,
    fragment_molecule,
)

rdBase.DisableLog("rdApp.error")


def save_json(path: str, arr: Any) -> None:
    with open(path, "w") as f:
        json.dump(arr, f)


def load_json(path: str) -> Any:
    with open(path) as f:
        out = json.load(f)
    return out


def save_pickle(path: str, arr: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(arr, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        out = pickle.load(f)
    return out


class FragDB:
    def __init__(self) -> None:
        # raw fragments
        self.frag_names: Dict[Any, Any] = {}
        self.frag_counts: List[int] = []
        self.frag_elem: List[Any] = []
        self.frag_r: List[Any] = []
        self.frag_symmgroups: Dict[Any, Any] = {}  # {frag_name: ig_table}

        # unique fragments
        self.ufrag_names: Dict[
            Any, Any
        ] = {}  # {ufrag_name: index} # ufrag_name = {frag_name}_{symmetry_group}
        self.ufrag_smis: List[Any] = []
        self.ufrag_counts: List[int] = []
        self.ufrag_stem: List[Any] = []
        self.ufrag_r: List[Any] = []

    def add_frag(self, frag_name: str, frag_elem: List[str], frag_r: List[int]) -> None:
        frag_natm = len(frag_elem)
        # if the fragment is new, initialize empty holders
        if frag_name not in self.frag_names:
            frag_idx = len(self.frag_names)
            self.frag_names[frag_name] = frag_idx
            self.frag_counts.append(0)
            self.frag_elem.append(frag_elem)
            self.frag_r.append(np.zeros([frag_natm, 4]))
        # add fragments
        frag_idx = self.frag_names[frag_name]
        self.frag_counts[frag_idx] += 1
        assert self.frag_elem[frag_idx] == frag_elem, (
            "fragment's order of elements does not repeat"
        )

        # add r-groups in a sparse format
        if len(frag_r) > 0:
            frag_r_updates = np.zeros([frag_natm, 4])
            for atmid, count in Counter(frag_r).items():
                frag_r_updates[atmid, count - 1] = 1
            self.frag_r[frag_idx] = self.frag_r[frag_idx] + frag_r_updates
        else:
            pass  # molecule consisted of a single fragment; no updates to the R-groups

    def add_frags(
        self, jun_bonds: np.ndarray, frag_names: List[str], frag_elem: List[List[str]]
    ) -> None:
        # for each of the groups find a list of fragment bonds, and R-groups
        frag_r = [[] for _ in range(len(jun_bonds) + 1)]  # type: ignore
        for bond in jun_bonds:
            frag_r[bond[0]].append(bond[2])
            frag_r[bond[1]].append(bond[3])
        nfrags = len(frag_names)
        for i in range(nfrags):
            self.add_frag(frag_names[i], frag_elem[i], frag_r[i])

    def add_mol(
        self, mol: Chem.Mol, frags_generic: bool, decomposition: str = "rota_murcko"
    ) -> Tuple[List[List[str]], np.ndarray, List[str]]:
        frag_elem, frag_coord, _, jun_bonds, frag_names = fragment_molecule(
            mol, frags_generic, decomposition=decomposition
        )

        self.add_frags(jun_bonds, frag_names, frag_elem)
        return frag_elem, jun_bonds, frag_names

    def make_ufrags(self, verbose: bool = False) -> None:
        "generate the new database of unique fragments"
        passed_nfrag = 0
        broken_nfrag = 0
        for frag_name in self.frag_names:
            # compute isometry groups of a fragment
            try:
                frag_mol = Chem.MolFromSmiles(frag_name)
                frag_symm = compute_isometry(frag_mol)
            except Exception as e:
                raise e
                # todo log errors
                broken_nfrag += 1
                continue
            passed_nfrag += 1
            self.frag_symmgroups[frag_name] = frag_symm

            # find (possible) bonds of a fragment
            frag_r = self.frag_r[self.frag_names[frag_name]]
            atm_bonded = frag_r.sum(1) > 0
            frag_symm_bonded = np.unique(frag_symm[atm_bonded])
            frag_natm = frag_r.shape[0]

            for symm_group in frag_symm_bonded:
                # add ufrag name
                ufrag_name = frag_name + "_" + str(symm_group)
                ufrag_idx = len(self.ufrag_names)
                self.ufrag_names[ufrag_name] = ufrag_idx
                self.ufrag_smis.append(frag_name)
                # ufrag count
                ufrag_count = frag_r[(frag_symm == symm_group)].sum(0)
                ufrag_count = (ufrag_count * np.array([1, 2, 3, 4])).sum()
                self.ufrag_counts.append(ufrag_count)
                # ufrag_r
                # todo: add all R-groups > count except that one
                stem = np.where(frag_symm == symm_group)[0][0]
                r_mask = np.ones(frag_natm)
                r_mask[stem] = 0.0
                self.ufrag_stem.append(stem)
                self.ufrag_r.append(frag_r * r_mask[:, None])

        total_ufrag = (
            np.concatenate(self.frag_r, axis=0) * np.array([[1, 2, 3, 4]])
        ).sum()
        print(
            "passed frags:",
            passed_nfrag,
            "broken frags:",
            broken_nfrag,
            "passed ufrag count",
            np.sum(self.ufrag_counts),
            "total ufrag count",
            total_ufrag,
        )
        if verbose:
            plt.plot(-np.sort(-np.asarray(self.ufrag_counts)))
            plt.yscale("log")
            plt.xlabel("ufrag id")
            plt.ylabel("ufrag count")
            plt.show()
        return None

    def filter_ufrags(
        self, block_cutoff: int = 250, r_cutoff: int = 100, verbose: bool = True
    ) -> Tuple[Tuple[str, ...], List[str], np.ndarray, np.ndarray]:
        "deletes most of the uni"
        # unique fragments
        sel_ufrag_names = {}
        sel_ufrag_smis = []
        sel_ufrag_counts = []  # type: ignore
        sel_ufrag_r = []
        for ufrag_name in self.ufrag_names:
            ufrag_idx = self.ufrag_names[ufrag_name]
            if self.ufrag_counts[ufrag_idx] >= block_cutoff:
                sel_ufrag_smis.append(self.ufrag_smis[ufrag_idx])
                sel_ufrag_idx = len(sel_ufrag_counts)
                sel_ufrag_names[ufrag_name] = sel_ufrag_idx
                sel_ufrag_counts.append(self.ufrag_counts[ufrag_idx])

                ufrag_r = self.ufrag_r[ufrag_idx]
                ufrag_r = np.fliplr(np.cumsum(np.fliplr(ufrag_r), axis=1))

                ufrag_r = np.concatenate(
                    [
                        np.array([self.ufrag_stem[ufrag_idx]]),
                        np.where(ufrag_r[:, 3] >= r_cutoff)[0],
                        np.where(ufrag_r[:, 2] >= r_cutoff)[0],
                        np.where(ufrag_r[:, 1] >= r_cutoff)[0],
                        np.where(ufrag_r[:, 0] >= r_cutoff)[0],
                    ]
                )

                sel_ufrag_r.append(ufrag_r)
        # reorder keys values alphabetically
        print(sel_ufrag_names)
        print(sel_ufrag_r)
        sel_ufrag_names, order = zip(*sel_ufrag_names.items())  # type: ignore
        sel_ufrag_counts = np.asarray(sel_ufrag_counts)[np.asarray(order)]
        sel_ufrag_r = [sel_ufrag_r[idx] for idx in np.asarray(order)]

        print(
            "num ufrag selected",
            len(sel_ufrag_counts),
            "total ufrag count",
            np.sum(self.ufrag_counts),
            "sel_ufrag_count",
            np.sum(sel_ufrag_counts),
        )
        print("selected ufrags:", sel_ufrag_names)
        return sel_ufrag_names, sel_ufrag_smis, sel_ufrag_counts, sel_ufrag_r  # type: ignore

    def get_ufrag(self, frag_name: str, frag_bond: int) -> Tuple[int, int]:
        frag_symm = self.frag_symmgroups[frag_name]
        symm_group = frag_symm[frag_bond]  # fixme should it be 4 ???
        ufrag_name = frag_name + "_" + str(symm_group)
        ufrag_idx = self.ufrag_names[ufrag_name]
        ufrag_count = self.ufrag_counts[ufrag_idx]
        return ufrag_idx, ufrag_count

    def get_mol_ufrag(
        self,
        mol: Chem.Mol,
        frags_generic: bool,
        decomposition: str = "rota_murcko",
        elem_asnum: bool = True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, List[int]]:
        # break molecule into fragments
        frag_elems, frag_coords, _, jun_bonds, frag_names = fragment_molecule(
            mol, frags_generic, decomposition
        )
        # find every fragment in the database
        ufrag_idxs = []
        flat_jun_bonds = np.concatenate(
            [
                np.stack([jun_bonds[:, 0], jun_bonds[:, 2]], 1),
                np.stack([jun_bonds[:, 1], jun_bonds[:, 3]], 1),
            ],
            axis=0,
        )
        for bond in flat_jun_bonds:
            ufrag_idxs.append(self.get_ufrag(frag_names[bond[0]], bond[1])[0])
        # convert elements to atomic numbers
        if elem_asnum:
            _frag_elems = []
            for frag_elem in frag_elems:
                _frag_elems.append(
                    np.asarray([atomic_numbers[e] for e in frag_elem], dtype=np.float32)
                )
            frag_elems = _frag_elems
        return frag_elems, frag_coords, jun_bonds, ufrag_idxs

    def save_state(self, fragdb_path: str) -> None:
        "write this object to the disk"
        if not os.path.exists(fragdb_path):
            os.makedirs(fragdb_path)
        nr = (
            len(self.frag_names),
            len(self.frag_counts),
            len(self.frag_elem),
            len(self.frag_r),
        )
        assert all(np.asarray(nr) == nr[0]), (
            "database columns are not of the same length"
        )
        np.save(os.path.join(fragdb_path, "frag_names.npy"), self.frag_names)
        np.save(os.path.join(fragdb_path, "frag_counts.npy"), self.frag_counts)
        save_json(os.path.join(fragdb_path, "frag_elem.json"), self.frag_elem)
        print(self.frag_r)
        save_pickle(os.path.join(fragdb_path, "frag_r.pkl"), self.frag_r)
        np.save(os.path.join(fragdb_path, "frag_symmgroups.npy"), self.frag_symmgroups)
        assert len(self.ufrag_names) == len(self.ufrag_counts), (
            "broken database of ufragments"
        )
        np.save(os.path.join(fragdb_path, "ufrag_names.npy"), self.ufrag_names)
        np.save(os.path.join(fragdb_path, "ufrag_counts.npy"), self.ufrag_counts)
        print("saved database state nfrag:", len(self.frag_names))

    def load_state(self, fragdb_path: str, ufrag: bool) -> None:
        "write this object to the disk"
        self.frag_names = np.load(
            os.path.join(fragdb_path, "frag_names.npy"), allow_pickle=True
        ).item()
        self.frag_counts = np.load(os.path.join(fragdb_path, "frag_counts.npy"))
        self.frag_elem = load_json(os.path.join(fragdb_path, "frag_elem.json"))
        self.frag_r = load_pickle(os.path.join(fragdb_path, "frag_r.pkl"))
        nr = (
            len(self.frag_names.keys()),
            len(self.frag_counts),
            len(self.frag_elem),
            len(self.frag_r),
        )
        assert all(np.asarray(nr) == nr[0]), (
            "database columns are not of the same length"
        )
        if ufrag:
            self.frag_symmgroups = np.load(
                os.path.join(fragdb_path, "frag_symmgroups.npy"), allow_pickle=True
            ).item()
            self.ufrag_names = np.load(
                os.path.join(fragdb_path, "ufrag_names.npy"), allow_pickle=True
            ).item()
            self.ufrag_counts = np.load(os.path.join(fragdb_path, "ufrag_counts.npy"))
            assert len(self.ufrag_names) == len(self.ufrag_counts), (
                "broken database of ufragments"
            )
        print("loaded database state nfrag:", len(self.frag_names))


class cfg:
    source_molfile = "data/properties.csv"
    fragdb_path = "data/fragdb"
    frags_generic = False
    block_cutoff = 100
    r_cutoff = 100


if __name__ == "__main__":
    fragdb = FragDB()

    def fragdb_from_molfile(molfile: str, maxmol: int = 10000000) -> None:
        df = pd.read_csv(molfile)[:]
        mols = [Chem.MolFromSmiles(smi) for smi in df["smiles"]]
        passed_nmols = 0
        broken_nmols = 0
        for mol in tqdm(mols):
            try:
                mol_frags = fragdb.add_mol(mol, frags_generic=cfg.frags_generic)
                if mol_frags is not None:
                    passed_nmols += 1
                else:
                    broken_nmols += 1
            except Exception as e:
                raise e
                broken_nmols += 1
            if passed_nmols > maxmol:
                break
        print("molecules passed:", passed_nmols, "molecules broken", broken_nmols)

    fragdb_from_molfile(cfg.source_molfile)
    # fragdb.save_state(cfg.fragdb_path)
    # fragdb.load_state(cfg.fragdb_path, ufrag=False)

    fragdb.make_ufrags()
    block_names, block_smis, _, block_rs = fragdb.filter_ufrags(
        verbose=False, block_cutoff=cfg.block_cutoff, r_cutoff=cfg.r_cutoff
    )

    blocks = pd.DataFrame(
        {"block_name": block_names, "block_smi": block_smis, "block_r": block_rs}
    )
    blocks.to_json("data/fragdb/example_blocks.json")

    print(pd.read_json("data/fragdb/example_blocks.json"))
