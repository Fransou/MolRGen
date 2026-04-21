"""Reward functions for molecular optimization."""

from typing import Any, List, Union

import networkx as nx
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, rdMolDescriptors
from rdkit.Chem.rdmolfiles import MolFromSmiles

try:
    from rdkit.Contrib.SA_Score import sascorer
except ModuleNotFoundError:
    # fix from https://github.com/rdkit/rdkit/issues/2279
    import os
    import sys

    sys.path.append(os.path.join(Chem.RDConfig.RDContribDir, "SA_Score"))
    import sascorer


def penalized_logp(mol: Chem.Mol) -> float:
    """Copy of TDC's penalized logP implementation."""
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455
    log_p: float = Descriptors.MolLogP(mol)
    SA: float = -sascorer.calculateScore(mol)

    # cycle score
    cycle_list: List[Any] = nx.cycle_basis(
        nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol))
    )
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p + normalized_SA + normalized_cycle


class RDKITOracle:
    """
    Class implementing all Rdkit descriptors.
    """

    def __init__(self, name: str):
        """Initialize RDKit descriptor oracle.

        Args:
            name: Name of the RDKit descriptor to use.

        Raises:
            ValueError: If the descriptor name is not found in RDKit.
        """
        self.name = name
        self.descriptor = self.get_descriptor()

    def get_descriptor(self) -> Any:
        """Get the descriptor from Rdkit."""
        if self.name.lower() == "qed":
            return QED.qed
        if self.name.lower() == "sa":
            return sascorer.calculateScore
        if self.name.lower() == "logp":
            return penalized_logp
        if hasattr(rdMolDescriptors, self.name):
            return getattr(rdMolDescriptors, self.name)
        raise ValueError(f"Descriptor {self.name} not found in Rdkit.")

    def __call__(self, smi_mol: Union[str, List[str]]) -> Union[float, List[float]]:
        """Get the descriptor value for the molecule."""
        if isinstance(smi_mol, list):
            if len(smi_mol) == 0:
                return []
            if isinstance(smi_mol[0], str):
                mols = [MolFromSmiles(smi) for smi in smi_mol]
            else:
                raise ValueError("Input must be a list of SMILES strings.")
            return [
                float(self.descriptor(mol)) if mol is not None else 0 for mol in mols
            ]

        if isinstance(smi_mol, str):
            mol = MolFromSmiles(smi_mol)
        else:
            raise ValueError("Input must be a SMILES string.")
        if mol is None:
            return 0
        return float(self.descriptor(mol))
