"""Tests for all molecular verification tasks via the server.

This module contains server-based tests for:
- Molecular property verification (regression, classification)
- Molecular reaction verification (full_path, final_product, reactants)
- Molecular generation verification (valid SMILES, property optimization)
"""

import logging
import time
from typing import Any, Dict, List

import pytest
import requests

from molrgen.server_utils.utils import MolecularVerifierServerResponse

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")  # type: ignore
def ensure_server(uvicorn_server: Any, server_url: str) -> str:
    """Ensure the server is available for tests."""
    try:
        response = requests.get(f"{server_url}/liveness", timeout=5)
        if response.status_code == 200:
            return server_url
    except requests.exceptions.RequestException:
        logger.warning(f"Failed to connect to {server_url}")

    pytest.skip("Server is not available")
    return ""


# =============================================================================
# Helper Functions
# =============================================================================


def get_rewards(
    server_url: str, completions: List[str], metadata: List[Dict[str, Any]]
) -> List[MolecularVerifierServerResponse]:
    """Get rewards from the server."""
    responses = []
    for comp, meta in zip(completions, metadata):
        response = requests.post(
            f"{server_url}/get_reward",
            json={"metadata": meta, "query": comp},
        )
        responses.append(MolecularVerifierServerResponse(**response.json()))
        time.sleep(0.1)  # Small delay to avoid overwhelming the server

    return responses


# =============================================================================
# Server Availability Tests
# =============================================================================


class TestServerAvailability:
    """Tests for server availability."""

    def test_server_health(self, ensure_server: str) -> None:
        """Test server liveness endpoint."""
        response = requests.get(f"{ensure_server}/liveness", timeout=5)
        assert response.status_code == 200


# =============================================================================
# Property Prediction Verification Server Tests
# =============================================================================


class TestPropertyPredictionServer:
    """Tests for property prediction (regression/classification) via server."""

    def test_regression_exact_match(self, ensure_server: str) -> None:
        """Regression: exact match should return 1.0."""
        responses = get_rewards(
            ensure_server,
            completions=["<answer> 0.5 </answer>"],
            metadata=[
                {
                    "objectives": ["regression"],
                    "properties": [""],
                    "target": [0.5],
                    "norm_var": 1.0,
                }
            ],
        )
        assert responses[0].reward == 1.0

    def test_regression_close_value(self, ensure_server: str) -> None:
        """Regression: close values should return high reward."""
        responses = get_rewards(
            ensure_server,
            completions=["<answer> 0.5001 </answer>"],
            metadata=[
                {
                    "objectives": ["regression"],
                    "properties": [""],
                    "target": [0.5],
                    "norm_var": 1.0,
                }
            ],
        )
        assert responses[0].reward > 0.9

    def test_regression_far_value(self, ensure_server: str) -> None:
        """Regression: far values should return low reward."""
        responses = get_rewards(
            ensure_server,
            completions=["<answer> 10.0 </answer>"],
            metadata=[
                {
                    "objectives": ["regression"],
                    "properties": [""],
                    "target": [0.5],
                    "norm_var": 1.0,
                }
            ],
        )
        assert responses[0].reward < 0.5

    def test_regression_scientific_notation(self, ensure_server: str) -> None:
        """Regression: scientific notation should be parsed correctly."""
        responses = get_rewards(
            ensure_server,
            completions=["<answer> 5e-1 </answer>"],
            metadata=[
                {
                    "objectives": ["regression"],
                    "properties": [""],
                    "target": [0.5],
                    "norm_var": 1.0,
                }
            ],
        )
        assert responses[0].reward == 1.0

    def test_classification_correct_1(self, ensure_server: str) -> None:
        """Classification: correct answer 1 should return 1.0."""
        responses = get_rewards(
            ensure_server,
            completions=["<answer> 1 </answer>"],
            metadata=[
                {
                    "objectives": ["classification"],
                    "properties": [""],
                    "target": [1],
                    "norm_var": 1.0,
                }
            ],
        )
        assert responses[0].reward == 1.0

    def test_classification_correct_0(self, ensure_server: str) -> None:
        """Classification: correct answer 0 should return 1.0."""
        responses = get_rewards(
            ensure_server,
            completions=["<answer> 0 </answer>"],
            metadata=[
                {
                    "objectives": ["classification"],
                    "properties": [""],
                    "target": [0],
                    "norm_var": 1.0,
                }
            ],
        )
        assert responses[0].reward == 1.0

    def test_classification_wrong(self, ensure_server: str) -> None:
        """Classification: wrong answer should return 0.0."""
        responses = get_rewards(
            ensure_server,
            completions=["<answer> 0 </answer>"],
            metadata=[
                {
                    "objectives": ["classification"],
                    "properties": [""],
                    "target": [1],
                    "norm_var": 1.0,
                }
            ],
        )
        assert responses[0].reward == 0.0

    def test_classification_invalid(self, ensure_server: str) -> None:
        """Classification: invalid answer should return 0.0."""
        responses = get_rewards(
            ensure_server,
            completions=["<answer> invalid_text </answer>"],
            metadata=[
                {
                    "objectives": ["classification"],
                    "properties": [""],
                    "target": [1],
                    "norm_var": 1.0,
                }
            ],
        )
        assert responses[0].reward == 0.0

    def test_boxed_format(self, ensure_server: str) -> None:
        """Boxed format should be parsed correctly."""
        responses = get_rewards(
            ensure_server,
            completions=["<answer> \\boxed{ 0.75 } </answer>"],
            metadata=[
                {
                    "objectives": ["regression"],
                    "properties": [""],
                    "target": [0.75],
                    "norm_var": 1.0,
                }
            ],
        )
        assert responses[0].reward == 1.0


# =============================================================================
# Reaction Verification Server Tests
# =============================================================================


class TestReactionVerificationServer:
    """Tests for reaction verification via server."""

    def test_final_product_correct(self, ensure_server: str) -> None:
        """Final product: correct SMILES should return 1.0."""
        target_smiles = "CCO"
        responses = get_rewards(
            ensure_server,
            completions=[f'<answer> {{"answer": "{target_smiles}"}} </answer>'],
            metadata=[
                {
                    "objectives": ["final_product"],
                    "properties": [""],
                    "target": [target_smiles],
                    "reactants": [["CC", "O"]],
                    "products": [target_smiles],
                    "smarts": [""],
                    "or_smarts": [""],
                }
            ],
        )
        assert responses[0].reward == 1.0

    def test_final_product_wrong(self, ensure_server: str) -> None:
        """Final product: wrong SMILES should return 0.0."""
        responses = get_rewards(
            ensure_server,
            completions=['<answer> {"answer": 152} </answer>'],
            metadata=[
                {
                    "objectives": ["final_product"],
                    "properties": [""],
                    "target": ["CCO"],
                    "reactants": [["CC", "O"]],
                    "products": ["CCO"],
                    "smarts": [""],
                    "or_smarts": [""],
                }
            ],
        )
        assert responses[0].reward == 0.0

    def test_final_product_invalid_json(self, ensure_server: str) -> None:
        """Final product: invalid JSON should return 0.0."""
        responses = get_rewards(
            ensure_server,
            completions=["<answer> {invalid json} </answer>"],
            metadata=[
                {
                    "objectives": ["final_product"],
                    "properties": [""],
                    "target": ["CCO"],
                    "reactants": [["CC", "O"]],
                    "products": ["CCO"],
                    "smarts": [""],
                    "or_smarts": [""],
                }
            ],
        )
        assert responses[0].reward == 0.0


# =============================================================================
# Generation Verification Server Tests
# =============================================================================


class TestGenerationVerificationServer:
    """Tests for molecular generation verification via server."""

    def test_valid_smiles_returns_reward(self, ensure_server: str) -> None:
        """Valid SMILES should return non-zero reward."""
        responses = get_rewards(
            ensure_server,
            completions=["<answer> \\boxed{ CCO } </answer>"],
            metadata=[
                {"objectives": ["maximize"], "properties": ["qed"], "target": [0.0]}
            ],
        )
        assert responses[0].reward > 0.0

    def test_invalid_smiles_returns_zero(self, ensure_server: str) -> None:
        """Invalid SMILES should return 0.0."""
        responses = get_rewards(
            ensure_server,
            completions=["<answer> \\boxed{ NOT_A_MOLECULE } </answer>"],
            metadata=[
                {"objectives": ["maximize"], "properties": ["qed"], "target": [0.0]}
            ],
        )
        assert responses[0].reward == 0.0

    def test_maximize_qed(self, ensure_server: str) -> None:
        """Maximize QED: drug-like molecule should score higher."""
        # Aspirin-like vs simple methane
        responses = get_rewards(
            ensure_server,
            completions=[
                "<answer> \\boxed{ CC(=O)Oc1ccccc1C(=O)O } </answer>",  # Aspirin
                "<answer> \\boxed{ C } </answer>",  # Methane
            ],
            metadata=[
                {"objectives": ["maximize"], "properties": ["qed"], "target": [0.0]},
                {"objectives": ["maximize"], "properties": ["qed"], "target": [0.0]},
            ],
        )
        assert responses[0].reward > responses[1].reward

    def test_minimize_molecular_weight(self, ensure_server: str) -> None:
        """Minimize MW: smaller molecule should score higher."""
        responses = get_rewards(
            ensure_server,
            completions=[
                "<answer> \\boxed{ CCC=O } </answer>",
                "<answer> \\boxed{ CCCCCCCCCCCCBrBr } </answer>",
            ],
            metadata=[
                {
                    "objectives": ["minimize"],
                    "properties": ["CalcExactMolWt"],
                    "target": [0.0],
                },
                {
                    "objectives": ["minimize"],
                    "properties": ["CalcExactMolWt"],
                    "target": [0.0],
                },
            ],
        )
        assert responses[0].reward > responses[1].reward

    def test_above_threshold(self, ensure_server: str) -> None:
        """Above threshold: high logP molecule should score 1.0."""
        responses = get_rewards(
            ensure_server,
            completions=["<answer> \\boxed{ CCCCCCCCCC } </answer>"],  # High logP
            metadata=[
                {"objectives": ["above"], "properties": ["logP"], "target": [2.0]}
            ],
        )
        assert responses[0].reward == 1.0

    def test_multiple_properties(self, ensure_server: str) -> None:
        """Multiple properties should all be evaluated."""
        responses = get_rewards(
            ensure_server,
            completions=["<answer> \\boxed{ CCCO } </answer>"],
            metadata=[
                {
                    "objectives": ["maximize", "maximize"],
                    "properties": ["qed", "SA"],
                    "target": [0.0, 0.0],
                }
            ],
        )
        assert responses[0].reward > 0.0


# =============================================================================
# Batch Tests
# =============================================================================


class TestBatchRequests:
    """Tests for batch requests to server."""

    def test_batch_mixed_tasks(self, ensure_server: str) -> None:
        """Batch with mixed task types."""
        responses = get_rewards(
            ensure_server,
            completions=[
                "<answer> 1 </answer>",  # Classification
                "<answer> 0.5 </answer>",  # Regression
                "<answer> \\boxed{ CCO } </answer>",  # Generation
            ],
            metadata=[
                {"objectives": ["classification"], "properties": [""], "target": [1]},
                {"objectives": ["regression"], "properties": [""], "target": [0.5]},
                {"objectives": ["maximize"], "properties": ["qed"], "target": [0.0]},
            ],
        )
        assert responses[0].reward == 1.0  # Classification correct
        assert responses[1].reward == 1.0  # Regression exact match
        assert responses[2].reward > 0.0  # Generation valid molecule

    def test_batch_all_valid_molecules(self, ensure_server: str) -> None:
        """Batch of valid molecules should all return positive rewards."""
        smiles = ["CCC", "CCC", "CCC", "CCCO", "C=CCCO"]
        responses = get_rewards(
            ensure_server,
            completions=[f"<answer> \\boxed{{ {s} }} </answer>" for s in smiles],
            metadata=[
                {"objectives": ["maximize"], "properties": ["qed"], "target": [0.0]}
            ]
            * len(smiles),
        )
        assert len(responses) == len(smiles)
        assert all(r.reward > 0.0 for r in responses)

    def test_batch_all_invalid_molecules(self, ensure_server: str) -> None:
        """Batch of invalid molecules should all return 0.0."""
        invalids = ["FAKE1", "FAKE2", "NOT_SMILES", "INVALID"]
        responses = get_rewards(
            ensure_server,
            completions=[f"<answer> \\boxed{{ {s} }} </answer>" for s in invalids],
            metadata=[
                {"objectives": ["maximize"], "properties": ["qed"], "target": [0.0]}
            ]
            * len(invalids),
        )
        assert len(responses) == len(invalids)
        assert all(r.reward == 0.0 for r in responses)
