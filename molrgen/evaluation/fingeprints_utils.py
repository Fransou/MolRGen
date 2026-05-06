"""Molecular fingerprint utilities for similarity calculations.

This module provides utilities for computing molecular fingerprints and similarity matrices.
It supports multiple fingerprint types (ECFP, MACCS, RDKit, Gobbi2D, Avalon) and computes
Tanimoto similarity between molecules for use in diversity-aware metrics.
"""

from typing import Callable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def fp_name_to_fn(
    fp_name: str,
) -> Callable[[Chem.Mol], DataStructs.cDataStructs.ExplicitBitVect]:
    """Convert a fingerprint name to a function that generates that fingerprint.

    This function returns a callable that takes an RDKit Mol object and returns
    the corresponding fingerprint. It supports several standard fingerprint types
    used in cheminformatics.

    Args:
        fp_name: Name of the fingerprint to use. Supported options:
            - "ecfp{diameter}-{nbits}": Extended Connectivity Fingerprint (Morgan).
              Examples: "ecfp2-1024", "ecfp4-1024", "ecfp6-2048".
            - "maccs": MACCS Keys fingerprint (166 bit).
            - "rdkit": RDKit fingerprint.
            - "Gobbi2d": Gobbi pharmacophore 2D fingerprint.
            - "Avalon": Avalon fingerprint.

    Returns:
        A callable function that takes an RDKit Mol object and returns
        an ExplicitBitVect fingerprint.

    Raises:
        ValueError: If the fingerprint name is not recognized or has invalid format.
        AssertionError: If the ECFP parameters don't match expected format.

    Example:
        ```python
        from molrgen.evaluation.fingeprints_utils import fp_name_to_fn
        from rdkit import Chem

        # Get the fingerprint function
        fp_fn = fp_name_to_fn("ecfp4-1024")

        # Generate fingerprints for molecules
        mol = Chem.MolFromSmiles("c1ccccc1")
        fp = fp_fn(mol)
        ```

    Notes:
        - ECFP diameter is specified as topological diameter (divided by 2 internally
          to get the radius for Morgan fingerprints)
        - All returned fingerprints are explicit bit vectors for similarity calculations
    """

    if fp_name.startswith("ecfp"):
        d = int(fp_name[4])
        n_bits = int(fp_name.split("-")[1])
        assert fp_name == f"ecfp{d}-{n_bits}", f"Invalid fingerprint name: {fp_name}"

        def fp_fn(mol: Chem.Mol) -> DataStructs.cDataStructs.ExplicitBitVect:
            return AllChem.GetMorganFingerprintAsBitVect(mol, d // 2, n_bits)

        return fp_fn
    elif fp_name == "maccs":

        def fp_fn(mol: Chem.Mol) -> DataStructs.cDataStructs.ExplicitBitVect:
            return Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol)

        return fp_fn
    elif fp_name == "rdkit":

        def fp_fn(mol: Chem.Mol) -> DataStructs.cDataStructs.ExplicitBitVect:
            return Chem.RDKFingerprint(mol)

        return fp_fn
    elif fp_name == "Gobbi2d":
        from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

        def fp_fn(mol: Chem.Mol) -> DataStructs.cDataStructs.ExplicitBitVect:
            return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)

        return fp_fn
    elif fp_name == "Avalon":
        from rdkit.Avalon import pyAvalonTools

        def fp_fn(mol: Chem.Mol) -> DataStructs.cDataStructs.ExplicitBitVect:
            return pyAvalonTools.GetAvalonFP(mol)

        return fp_fn
    else:
        raise ValueError(f"Unknown fingerprint name: {fp_name}")


def get_sim_matrix(
    mols: list[Chem.Mol],
    fingerprint_name: str = "ecfp4-1024",
) -> np.ndarray[float]:
    """Compute a pairwise similarity matrix for a list of molecules.

    This function computes Tanimoto similarities between all pairs of molecules
    using the specified fingerprint method. The result is a condensed distance
    matrix (upper triangle only) in NumPy array format.

    Args:
        mols: List of RDKit Mol objects.
        fingerprint_name: Name of the fingerprint to use for similarity calculation.
            Default is "ecfp4-1024" (ECFP with diameter 4 and 1024 bits).
            See fp_name_to_fn for supported options.

    Returns:
        A 1D NumPy array containing the upper triangle of the pairwise similarity
        matrix. The array is in condensed form (as used by scipy.spatial.distance.squareform).
        For n molecules, the length is n*(n-1)/2.

    Example:
        ```python
        from molrgen.evaluation.fingeprints_utils import get_sim_matrix
        from rdkit import Chem

        smiles = ["c1ccccc1", "CC(C)Cc1ccc(cc1)C(C)C(O)=O", "CCO"]
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]

        sim_matrix = get_sim_matrix(mols, fingerprint_name="ecfp4-1024")
        print(f"Shape: {sim_matrix.shape}")  # (3,) for 3 molecules
        ```

    Notes:
        - The returned array is in condensed form. Use scipy.spatial.distance.squareform
          to convert to a full square similarity matrix.
        - Similarity values range from 0.0 (completely dissimilar) to 1.0 (identical)
        - The matrix is symmetric, so only the upper triangle is computed
    """
    fp_fn = fp_name_to_fn(fingerprint_name)
    fps = [fp_fn(mol) for mol in mols]
    sim_mat = [
        np.array(DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1 :]))
        for i, fp in enumerate(fps[:-1])
    ]
    matrix: np.ndarray[float] = np.concatenate(sim_mat)
    return matrix


def get_csim_matrix(
    molsA: list[Chem.Mol],
    molsB: list[Chem.Mol],
    fingerprint_name: str = "ecfp4-1024",
) -> np.ndarray[float]:
    """Compute a pairwise similarity matrix between two sets of molecules.

    This function computes Tanimoto similarities between all pairs of molecules
    from two different sets using the specified fingerprint method. The result is
    a 2D NumPy array where entry (i, j) contains the similarity between molsA[i]
    and molsB[j].

    Args:
        molsA: List of RDKit Mol objects for the first set.
        molsB: List of RDKit Mol objects for the second set.
        fingerprint_name: Name of the fingerprint to use for similarity calculation.
            Default is "ecfp4-1024" (ECFP with diameter 4 and 1024 bits).
            See fp_name_to_fn for supported options.

    Returns:
        A 2D NumPy array of shape (len(molsA), len(molsB)) containing the pairwise
        similarity values between molecules in molsA and molsB.
    """
    fp_fn = fp_name_to_fn(fingerprint_name)
    fpsA = [fp_fn(mol) for mol in molsA]
    fpsB = [fp_fn(mol) for mol in molsB]
    sim_mat = np.zeros((len(molsA), len(molsB)), dtype=float)
    for i, fpA in enumerate(fpsA):
        sim_mat[i, :] = np.array(DataStructs.BulkTanimotoSimilarity(fpA, fpsB))
    return sim_mat
