"""Tests for the RL Rewards generation server functionality."""

import logging
import os
import time
from typing import Any, Dict, List

import numpy as np
import pytest
import ray
import requests
import torch
from rdkit import Chem

from molrgen.server_utils.utils import MolecularVerifierServerResponse
from molrgen.utils.property_utils import has_bridged_bond

from ..utils import (
    compare_obj_reward_to_max,
    get_unscaled_obj,
    is_reward_valid,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Ray Initialization
# =============================================================================

if not ray.is_initialized():
    if os.environ.get("RAY_NAMESPACE") is not None:
        logger.warning(
            f"Initializing Ray with namespace: {os.environ.get('RAY_NAMESPACE')}"
        )
        ray.init(address="auto", namespace=os.environ.get("RAY_NAMESPACE"))
    else:
        ray.init()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session", params=list(range(4)))  # type:ignore
def idx_smiles(request: pytest.FixtureRequest) -> int:
    """
    Pytest fixture returning an index of a SMILES string.

    Example:
        def test_example(idx_smiles: int) -> None:
            ...
    """
    out: int = request.param
    return out


@pytest.fixture(scope="module")  # type:ignore
def ensure_server(uvicorn_server: Any, server_url: str) -> str:
    """
    Ensure the server is available for tests.

    This fixture depends on uvicorn_server to ensure automatic server management
    when --start-server is passed. It also checks if server is reachable.

    Args:
        uvicorn_server: The server process fixture from conftest.
        server_url: The server URL fixture from conftest.

    Returns:
        The server URL if available.

    Raises:
        pytest.skip: If server is not available.
    """
    try:
        response = requests.get(f"{server_url}/liveness", timeout=5)
        if response.status_code == 200:
            return server_url
    except requests.exceptions.RequestException:
        logger.warning(f"Failed to connect to {server_url}")
        pass

    pytest.skip("Generation reward server is not available")
    return ""


# =============================================================================
# Helper Functions
# =============================================================================


def get_rewards_async(
    server_url: str, completions: List[str], metadata: List[Dict[str, Any]]
) -> List[MolecularVerifierServerResponse]:
    """
    Get rewards asynchronously from the server for multiple completions.
    """

    @ray.remote(num_cpus=1)
    def get_reward_async(
        server_url: str, completions: str, metadata: Dict[str, Any]
    ) -> MolecularVerifierServerResponse:
        """
        Remote function to get reward from the server.

        Args:
            server_url: The server URL.
            smi: SMILES string to evaluate.
            metadata: Metadata dictionary containing properties, objectives, and target.

        Returns:
            Response from the reward server.
        """
        time.sleep(np.random.random() ** 2 * 0.2)
        r = requests.post(
            f"{server_url}/get_reward",
            json={
                "metadata": metadata,
                "query": completions,
                "prompts": "",
            },
        )
        return MolecularVerifierServerResponse(**r.json())

    return ray.get(
        [
            get_reward_async.remote(server_url, comp, meta)
            for comp, meta in zip(completions, metadata)
        ]
    )


# =============================================================================
# Server Availability Tests
# =============================================================================


class TestServerAvailability:
    """Tests for server availability checks."""

    def test_server_health_endpoint(self, ensure_server: str) -> None:
        """Test that the server liveness endpoint is accessible."""
        response = requests.get(f"{ensure_server}/liveness", timeout=5)
        assert response.status_code == 200


# =============================================================================
# Valid SMILES Server Tests
# =============================================================================


class TestValidSmilesServer:
    """Tests for valid SMILES reward scoring via the server."""

    def test_valid_smiles_with_fake_molecule(
        self,
        ensure_server: str,
    ) -> None:
        """Test that fake SMILES return zero reward."""
        completions = "<answer> FAKE </answer>"
        metadata = {
            "properties": ["qed"],
            "objectives": ["maximize"],
            "target": [0.0],
        }

        response = get_rewards_async(
            server_url=ensure_server, completions=[completions], metadata=[metadata]
        )
        rewards = np.array([r.reward for r in response])
        assert rewards.sum() == 0.0

    def test_valid_smiles_with_real_molecule(
        self,
        ensure_server: str,
    ) -> None:
        """Test that valid SMILES return non-zero reward."""
        valid_smiles = "CCC"
        completions = (
            f"Here is a molecule: <answer> \\boxed{{ {valid_smiles} }} </answer>"
        )
        metadata = {
            "properties": ["qed"],
            "objectives": ["maximize"],
            "target": [0.0],
        }

        response = get_rewards_async(
            server_url=ensure_server, completions=[completions], metadata=[metadata]
        )[0]
        reward = np.array(response.reward)
        assert reward > 0.0


# =============================================================================
# Multi-Prompt Multi-Generation Server Tests
# =============================================================================


class TestMultiPromptMultiGenerationServerAsync:
    """Tests for multiple prompts with multiple generations via the server."""

    def test_single_molecule_single_property(
        self,
        ensure_server: str,
        completion_smile: tuple[str, str],
    ) -> None:
        """Test reward calculation for a single molecule with a single property."""
        prop = "QED"
        completion, smiles = completion_smile
        metadata = {"properties": [prop], "objectives": ["maximize"], "target": [0]}

        response = get_rewards_async(
            server_url=ensure_server, completions=[completion], metadata=[metadata]
        )[0]
        rewards = response.reward

        is_smi_valid = Chem.MolFromSmiles(smiles) is not None and not has_bridged_bond(
            Chem.MolFromSmiles(smiles)
        )
        is_pattern_valid = not completion.startswith("[N]")

        if not is_pattern_valid or not is_smi_valid:
            # Invalid molecule should have zero reward
            assert rewards == 0.0
        else:
            is_reward_valid([rewards], [smiles], [prop])

    def test_multi_prompt_multiple_smiles(
        self,
        ensure_server: str,
        completions_smiles: tuple[List[str], List[List[str]]],
    ) -> None:
        """Test the reward function for multiple prompts with multiple SMILES each."""
        completions, smiles_chunks = completions_smiles
        properties = ["QED", "logP", "SA", "CalcNumHBA"][: len(completions)]

        # Send requests in parallel using Ray with different properties
        metadatas = [
            {
                "properties": [p],
                "objectives": ["maximize"],
                "target": [0],
            }
            for p in properties
        ]

        responses = get_rewards_async(
            server_url=ensure_server, completions=completions, metadata=metadatas
        )

        for i, (response, smiles_list, completion) in enumerate(
            zip(responses, smiles_chunks, completions)
        ):
            if completion.startswith("[N]"):
                continue
            meta_r = response.meta
            assert meta_r.generation_verifier_metadata is not None
            reward = meta_r.generation_verifier_metadata.all_smi_rewards
            smiles_v = [
                smi
                for smi in smiles_list
                if Chem.MolFromSmiles(smi) is not None
                and not has_bridged_bond(Chem.MolFromSmiles(smi))
            ]
            assert meta_r.generation_verifier_metadata.all_smi is not None
            assert set(smiles_v) == set(meta_r.generation_verifier_metadata.all_smi)

            if len(smiles_v) > 0:
                # Invalid molecule should have zero reward
                assert reward is not None
                is_reward_valid(reward, smiles_v, [properties[i]])


# =============================================================================
# Objective-Based Reward Server Tests
# =============================================================================


class TestObjectiveBasedRewardsServerAsync:
    """Tests for different optimization objectives via the server."""

    def test_all_objectives(
        self,
        ensure_server: str,
        completions_smile: tuple[List[str], List[str]],
        objective_to_test: str,
        prop: str,
    ) -> None:
        """
        Test the reward function with various optimization objectives.

        Assumes the value of the reward function when using maximize is correct.
        """
        completions, smiles_chunks = completions_smile
        obj_func, target = get_unscaled_obj(objective_to_test, prop)

        responses_obj = get_rewards_async(
            server_url=ensure_server,
            completions=completions,
            metadata=[
                {"properties": [prop], "objectives": [obj_func], "target": [target]}
            ]
            * len(completions),
        )

        responses_max = get_rewards_async(
            server_url=ensure_server,
            completions=completions,
            metadata=[
                {"properties": [prop], "objectives": ["maximize"], "target": [target]}
            ]
            * len(completions),
        )

        rewards_obj = [r.reward for r in responses_obj]
        rewards_max = [r.reward for r in responses_max]

        assert len(rewards_obj) == len(completions)
        assert len(rewards_max) == len(completions)

        mask = []
        for completion, smiles in zip(completions, smiles_chunks):
            if completion.startswith("[N]"):
                mask.append(False)
            else:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None or has_bridged_bond(mol):
                    mask.append(False)
                else:
                    mask.append(True)
        mask = torch.tensor(mask).float()
        rewards_obj_tensor = torch.Tensor(rewards_obj)
        rewards_max_tensor = torch.Tensor(rewards_max)
        assert not rewards_obj_tensor.isnan().any()

        objective = obj_func.split()[0]
        expected_val = compare_obj_reward_to_max(
            objective, target, rewards_max_tensor, mask, prop
        )
        assert torch.isclose(rewards_obj_tensor, expected_val, atol=1e-4).all()

    def test_maximize_objective(
        self,
        ensure_server: str,
        prop: str,
        completions_smiles: tuple[List[str], List[List[str]]],
    ) -> None:
        """Test the maximize objective specifically."""
        completions, smiles_batch = completions_smiles

        responses = get_rewards_async(
            server_url=ensure_server,
            completions=completions,
            metadata=[{"properties": [prop], "objectives": ["maximize"], "target": [0]}]
            * len(completions),
        )

        assert len(responses) == len(completions)

        for response, completion, smiles in zip(responses, completions, smiles_batch):
            reward = response.reward
            meta_r = response.meta

            smi_valid = [
                Chem.MolFromSmiles(smi) is not None
                and not has_bridged_bond(Chem.MolFromSmiles(smi))
                for smi in smiles
            ]
            pattern_valid = not completion.startswith("[N]")
            if sum(smi_valid) == 0 or not pattern_valid:
                assert reward == 0.0
            else:
                smiles_v = [smi for smi, valid in zip(smiles, smi_valid) if valid]
                assert meta_r.generation_verifier_metadata is not None
                assert set(smiles_v) == set(meta_r.generation_verifier_metadata.all_smi)
                all_smi_rewards = meta_r.generation_verifier_metadata.all_smi_rewards
                is_reward_valid(all_smi_rewards, smiles_v, [prop])
