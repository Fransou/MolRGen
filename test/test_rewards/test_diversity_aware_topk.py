from typing import Any

import numpy as np
import pytest
from rdkit import Chem
from scipy.spatial.distance import squareform

from molrgen.evaluation.diversity_aware_top_k import (
    div_aware_top_k_from_dist,
    diversity_aware_top_k,
)
from molrgen.evaluation.fingeprints_utils import get_sim_matrix

TEST_DIV_AWARE_TOPK_FROM_DIST = [
    {
        "dist": squareform(
            np.array(
                [
                    [0.0, 0.5, 0.9, 0.4],
                    [0.5, 0.0, 0.2, 0.8],
                    [0.9, 0.2, 0.0, 0.3],
                    [0.4, 0.8, 0.3, 0.0],
                ]
            )
        ),
        "weights": np.array([0.1, 0.4, 0.3, 0.2]),
        "results": {
            (1, 0.6): [1],
            (2, 0.6): [1, 3],
            (1, 0.1): [1],
            (2, 0.1): [1, 2],
            (3, 0.1): [1, 2, 3],
            (4, 0.1): [1, 2, 3, 0],
            (2, 0.25): [1, 3],
            (3, 0.25): [1, 3, 0],
            (4, 0.25): [1, 3, 0],
        },
    },
    {
        "dist": squareform(
            np.array(
                [
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.3],
                    [0.1, 0.0, 0.1, 0.1, 0.1, 0.1],
                    [0.2, 0.1, 0.0, 0.8, 0.5, 0.2],
                    [0.3, 0.1, 0.8, 0.0, 0.1, 0.6],
                    [0.4, 0.1, 0.5, 0.1, 0.0, 0.04],
                    [0.3, 0.1, 0.2, 0.6, 0.04, 0.0],
                ]
            )
        ),
        "weights": np.array([10, 5, 1, 7, 8, 9]),
        "results": {
            (1, 0.6): [0],
            (2, 0.6): [0],
            (1, 0.31): [0],
            (2, 0.31): [0, 4],
            (2, 0.21): [0, 5],
            (4, 0.21): [0, 5, 3],
            (2, 0.11): [0, 5],
            (3, 0.11): [0, 5, 3],
            (4, 0.05): [0, 5, 3, 1],
        },
    },
]


@pytest.mark.parametrize("data", TEST_DIV_AWARE_TOPK_FROM_DIST)  # type: ignore
def test_div_aware_top_k_from_dist(data: dict[str, Any]) -> None:
    for k, t in data["results"].keys():
        results = div_aware_top_k_from_dist(data["dist"], data["weights"], k=k, t=t)
        gt = data["results"][(k, t)]
        assert len(results) == len(gt) and all(results == gt), (
            f"Error with k,t={k, t}:\nresult: {results}\nexpected: {gt}"
        )


TEST_DIV_AWARE_TOPK = [
    {
        "k": 2,
        "t": 0.5,
        "mols": ["CCC", "CC", "CC(C)C", "CCNC", "CCNC"],
        "scores": [0.9, 0.6, 0.8, 1, 1],
        "expected": 0.95,
    },
    {
        "k": 2,
        "t": 0.3,
        "mols": ["CCC", "CC", "CC(C)C", "CCNC", "CCNC"],
        "scores": [0.9, 0.6, 0.8, 1, 1],
        "expected": 0.9,
    },
    {
        "k": 3,
        "t": 0.1,
        "mols": ["CCC", "CC", "CC(C)C", "CCNC", "CCNC"],
        "scores": [0.9, 0.6, 0.8, 1, 1],
        "expected": 0.6,
    },
    {
        "k": 3,
        "t": 0.3,
        "mols": ["CCC", "CC", "CC(C)C", "CCNC", "CCNC"],
        "scores": [0.9, 0.6, 0.8, 1, 1],
        "expected": 0.8,
    },
]


@pytest.mark.parametrize("data", TEST_DIV_AWARE_TOPK)  # type: ignore
def test_div_aware_top_k(data: dict[str, Any]) -> None:
    mols = [Chem.MolFromSmiles(smi) for smi in data["mols"]]
    sim_mat = squareform(get_sim_matrix(mols, fingerprint_name="ecfp4-128"))
    sim_mat = sim_mat + np.eye(len(mols))  # to make sure diagonal is 1.0
    results = [
        diversity_aware_top_k(
            mols=x,
            scores=data["scores"],
            k=data["k"],
            t=data["t"],
            fingerprint_name="ecfp4-128",
        )
        for x in (data["mols"], mols, sim_mat)
    ]
    assert all(np.isclose(res, data["expected"]) for res in results)
