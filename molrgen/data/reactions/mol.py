from functools import cached_property
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


class Molecule:
    def __init__(self, smiles: str) -> None:
        self._smiles = smiles.strip()

    @classmethod
    def from_rdmol(cls, rdmol: Chem.Mol) -> "Molecule":
        return cls(Chem.MolToSmiles(rdmol))

    def __getstate__(self) -> str:
        return self._smiles

    def __setstate__(self, state: str) -> None:
        self._smiles = state

    @property
    def smiles(self) -> str:
        return self._smiles

    @cached_property
    def _rdmol(self) -> Chem.Mol:
        return Chem.MolFromSmiles(self._smiles)

    @cached_property
    def _rdmol_no_hs(self) -> Chem.Mol:
        return Chem.RemoveHs(self._rdmol)

    @cached_property
    def is_valid(self) -> bool:
        return self._rdmol is not None

    @cached_property
    def csmiles(self) -> str:
        smiles = Chem.MolToSmiles(self._rdmol, canonical=True, isomericSmiles=False)
        assert isinstance(smiles, str)
        return smiles

    @cached_property
    def num_atoms(self) -> int:
        n_atoms = self._rdmol.GetNumAtoms()
        assert isinstance(n_atoms, int)
        return n_atoms

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Molecule) and self.csmiles == __value.csmiles

    def __hash__(self) -> int:
        return hash(self._smiles)

    @cached_property
    def fp(self) -> Any:
        return gen.GetFingerprint(self._rdmol)

    def tanimoto_similarity(self, other: list["Molecule"]) -> list[float]:
        similarities: list[float] = DataStructs.BulkTanimotoSimilarity(
            self.fp, [mol.fp for mol in other]
        )
        return similarities

    def inters_similarity(self, other: list["Molecule"]) -> list[float]:
        similarities: list[float] = DataStructs.BulkTanimotoSimilarity(
            self.fp, [mol.fp for mol in other]
        )
        return similarities
