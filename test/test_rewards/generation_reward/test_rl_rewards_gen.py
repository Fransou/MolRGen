"""Tests for the RL Rewards generation functionality."""

from typing import List

import numpy as np
import pytest
import torch
from rdkit import Chem

from molrgen.reward import (
    GenerationVerifierConfigModel,
    MolecularVerifier,
    MolecularVerifierConfigModel,
)
from molrgen.utils.property_utils import (
    has_bridged_bond,
)

from ..utils import (
    compare_obj_reward_to_max,
    get_unscaled_obj,
    is_reward_valid,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")  # type:ignore
def valid_scorer(data_path: str) -> MolecularVerifier:
    """Create a RewardScorer for valid SMILES checking."""
    return MolecularVerifier(
        MolecularVerifierConfigModel(
            parsing_method="answer_tags",
            reward="valid_smiles",
            generation_verifier_config=GenerationVerifierConfigModel(
                path_to_mappings=data_path
            ),
        )
    )


@pytest.fixture(scope="module")  # type:ignore
def property_scorer(data_path: str) -> MolecularVerifier:
    """Create a RewardScorer for property scoring without rescaling."""
    return MolecularVerifier(
        MolecularVerifierConfigModel(
            parsing_method="answer_tags",
            reward="property",
            generation_verifier_config=GenerationVerifierConfigModel(
                path_to_mappings=data_path
            ),
        )
    )


# =============================================================================
# Valid SMILES Tests
# =============================================================================


class TestValidSmiles:
    """Tests for valid SMILES reward scoring."""

    def test_valid_smiles_with_fake_molecule(
        self,
        valid_scorer: MolecularVerifier,
    ) -> None:
        """Test that fake SMILES return zero reward."""
        completions = ["<answer> FAKE </answer>"]
        rewards = np.array(
            valid_scorer(
                completions,
                metadata=[
                    {
                        "objectives": ["maximize"],
                        "properties": ["QED"],
                        "target": [0.0],
                    }
                ],
            ).rewards
        )
        assert rewards.sum() == 0.0

    def test_valid_smiles_with_real_molecule(
        self,
        valid_scorer: MolecularVerifier,
    ) -> None:
        """Test that valid SMILES return non-zero reward."""
        valid_smiles = "CCC"
        completions = [
            f"Here is a molecule: <answer> \\boxed{{ {valid_smiles} }} </answer>"
        ]

        rewards = np.array(
            valid_scorer(
                completions,
                metadata=[
                    {
                        "objectives": ["maximize"],
                        "properties": ["QED"],
                        "target": [0.0],
                    }
                ],
            ).rewards
        )
        assert rewards.sum() == 1.0

    def test_valid_smiles_scoring_single(
        self,
        valid_scorer: MolecularVerifier,
        completion_smile: tuple[str, str],
    ) -> None:
        """Test the valid SMILES reward function with a single SMILES."""
        completion, smiles = completion_smile
        completions = [completion]

        rewards = np.array(
            valid_scorer(
                completions,
                metadata=[
                    {
                        "objectives": ["maximize"],
                        "properties": ["QED"],
                        "target": [0.0],
                    }
                ],
            ).rewards
        )

        # Check if completion pattern is valid (doesn't start with [N])
        # and if the SMILES produces a valid molecule without bridged bonds
        mol = Chem.MolFromSmiles(smiles)
        is_valid_pattern = not completion.startswith("[N]")
        is_valid_mol = mol is not None and not has_bridged_bond(mol)
        expected = float(is_valid_pattern and is_valid_mol)

        assert rewards.sum() == expected

    def test_valid_smiles_scoring_multiple_in_one(
        self,
        valid_scorer: MolecularVerifier,
        completion_smiles: tuple[str, list[str]],
    ) -> None:
        """Test the valid SMILES reward function with multiple SMILES in one completion."""
        completion, smis = completion_smiles
        completions = [completion]

        rewards = np.array(
            valid_scorer(
                completions,
                metadata=[
                    {
                        "objectives": ["maximize"],
                        "properties": ["QED"],
                        "target": [0.0],
                    }
                ],
            ).rewards
        )

        are_smi_valid = [
            int(
                Chem.MolFromSmiles(smi) is not None
                and not has_bridged_bond(Chem.MolFromSmiles(smi))
            )
            for smi in smis
        ]
        if sum(are_smi_valid) != 1 or completion.startswith("[N]"):
            assert rewards.sum() == 0.0
        else:
            assert rewards.sum() == 1.0

    def test_valid_smiles_scoring_batch_single_smiles(
        self,
        valid_scorer: MolecularVerifier,
        completions_smile: tuple[List[str], List[str]],
    ) -> None:
        """Test the valid SMILES reward function with a batch of completions, each with one SMILES."""
        completions, smiles_list = completions_smile

        rewards = np.array(
            valid_scorer(
                completions,
                metadata=[
                    {
                        "objectives": ["maximize"],
                        "properties": ["QED"],
                        "target": [0.0],
                    }
                ]
                * len(completions),
            ).rewards
        )

        assert len(rewards) == len(completions)

        # Each reward should be 0 or 1
        for i, (completion, smi) in enumerate(zip(completions, smiles_list)):
            mol = Chem.MolFromSmiles(smi)
            is_valid_pattern = not completion.startswith("[N]")
            is_valid_mol = mol is not None and not has_bridged_bond(mol)
            expected = float(is_valid_pattern and is_valid_mol)
            assert rewards[i] == expected, f"Mismatch at index {i}: {completion}"

    def test_valid_smiles_scoring_batch_multiple_smiles(
        self,
        valid_scorer: MolecularVerifier,
        completions_smiles: tuple[List[str], List[List[str]]],
    ) -> None:
        """Test the valid SMILES reward function with a batch of completions, each with multiple SMILES."""
        completions, smiles_chunks = completions_smiles

        rewards = np.array(
            valid_scorer(
                completions,
                metadata=[
                    {
                        "objectives": ["maximize"],
                        "properties": ["QED"],
                        "target": [0.0],
                    }
                ]
                * len(completions),
            ).rewards
        )

        assert len(rewards) == len(completions)

        # Each reward should be between 0 and 1
        for reward, smiles, completion in zip(rewards, smiles_chunks, completions):
            are_smi_valid = [
                int(
                    Chem.MolFromSmiles(smi) is not None
                    and not has_bridged_bond(Chem.MolFromSmiles(smi))
                )
                for smi in smiles
            ]
            if sum(are_smi_valid) == 1 and not completion.startswith("[N]"):
                assert reward == 1.0
            else:
                assert reward == 0.0


# =============================================================================
# Multi-Prompt Multi-Generation Tests
# =============================================================================


class TestMultiPromptMultiGeneration:
    """Tests for multiple prompts with multiple generations."""

    def test_single_molecule_single_property(
        self,
        property_scorer: MolecularVerifier,
        completion_smile: tuple[str, str],
    ) -> None:
        """Test reward calculation for a single molecule with a single property."""
        prop = "QED"
        completion, smiles = completion_smile
        metadata = [{"properties": [prop], "objectives": ["maximize"], "target": [0]}]
        output = property_scorer([completion], metadata)
        rewards = output.rewards
        assert len(rewards) == 1
        is_smi_valid = Chem.MolFromSmiles(smiles) is not None and not has_bridged_bond(
            Chem.MolFromSmiles(smiles)
        )
        is_pattern_valid = not completion.startswith("[N]")

        if not is_pattern_valid or not is_smi_valid:
            # Invalid molecule should have zero reward
            assert rewards[0] == 0.0
        else:
            is_reward_valid(rewards, [smiles], [prop])

    def test_multi_prompt_single_smiles(
        self,
        property_scorer: MolecularVerifier,
        completions_smile: tuple[List[str], List[str]],
    ) -> None:
        """Test the reward function for multiple prompts with single SMILES each."""
        prop = "QED"
        completions, smiles_list = completions_smile

        metadata = [
            {"properties": [prop], "objectives": ["maximize"], "target": [0]}
            for _ in range(len(completions))
        ]

        output = property_scorer(completions, metadata)
        rewards = output.rewards

        assert len(rewards) == len(completions)
        for i, (reward, smi, completion) in enumerate(
            zip(rewards, smiles_list, completions)
        ):
            is_smi_valid = Chem.MolFromSmiles(smi) is not None and not has_bridged_bond(
                Chem.MolFromSmiles(smi)
            )
            is_pattern_valid = not completion.startswith("[N]")
            if not is_pattern_valid or not is_smi_valid:
                assert reward == 0.0
            else:
                is_reward_valid([reward], [smi], [prop])

    def test_multi_prompt_multiple_smiles(
        self,
        property_scorer: MolecularVerifier,
        completions_smiles: tuple[List[str], List[List[str]]],
    ) -> None:
        """Test the reward function for multiple prompts with multiple SMILES each."""
        completions, smiles_chunks = completions_smiles
        properties = ["QED", "logP", "SA", "CalcNumHBA"][: len(completions)]
        metadata = [
            {"properties": [p], "objectives": ["maximize"], "target": [0]}
            for p in properties
        ]
        output = property_scorer(completions, metadata)
        metas_r = output.verifier_metadatas
        for i, (meta, smiles_list, completion) in enumerate(
            zip(metas_r, smiles_chunks, completions)
        ):
            if completion.startswith("[N]"):
                continue
            assert meta.generation_verifier_metadata is not None
            reward = meta.generation_verifier_metadata.all_smi_rewards
            smiles_v = [
                smi
                for smi in smiles_list
                if Chem.MolFromSmiles(smi) is not None
                and not has_bridged_bond(Chem.MolFromSmiles(smi))
            ]
            if len(smiles_v) > 0:
                # Invalid molecule should have zero reward
                is_reward_valid(reward, smiles_v, [properties[i]])


# =============================================================================
# Objective-Based Reward Tests
# =============================================================================


class TestObjectiveBasedRewards:
    """Tests for different optimization objectives."""

    def test_all_objectives(
        self,
        property_scorer: MolecularVerifier,
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
        metadata = [
            {"properties": [prop], "objectives": [obj_func], "target": [target]}
        ] * len(completions) + [
            {
                "properties": [prop],
                "objectives": ["maximize"],
                "target": [target],
            }
        ] * len(completions)

        output = property_scorer(completions + completions, metadata)
        rewards = output.rewards
        assert len(rewards) == len(completions) * 2
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
        rewards_tensor = torch.Tensor(rewards)
        assert not rewards_tensor.isnan().any()

        rewards_prop = rewards_tensor[: len(completions)]
        rewards_max = rewards_tensor[len(completions) :]

        objective = obj_func.split()[0]
        expected_val = compare_obj_reward_to_max(
            objective, target, rewards_max, mask, prop
        )
        assert torch.isclose(rewards_prop, expected_val, atol=1e-4).all()

    def test_maximize_objective(
        self,
        property_scorer: MolecularVerifier,
        prop: str,
        completions_smiles: tuple[List[str], List[List[str]]],
    ) -> None:
        """Test the maximize objective specifically."""
        completions, smiles_batch = completions_smiles
        metadata = [
            {"properties": [prop], "objectives": ["maximize"], "target": [0]}
            for _ in smiles_batch
        ]
        output = property_scorer(completions, metadata)
        rewards = output.rewards
        meta_rs = output.verifier_metadatas
        assert len(rewards) == len(completions)

        for reward, meta_r, completion, smiles in zip(
            rewards, meta_rs, completions, smiles_batch
        ):
            smi_valid = [
                Chem.MolFromSmiles(smi) is not None
                and not has_bridged_bond(Chem.MolFromSmiles(smi))
                for smi in smiles
            ]
            pattern_valid = not completion.startswith("[N]")
            if sum(smi_valid) == 0 or not pattern_valid:
                assert reward == 0.0
            else:
                smis_v = [smi for smi, valid in zip(smiles, smi_valid) if valid]
                assert meta_r.generation_verifier_metadata is not None
                all_smi_rewards = meta_r.generation_verifier_metadata.all_smi_rewards
                is_reward_valid(all_smi_rewards, smis_v, [prop])


# =============================================================================
# Bridged Bond Tests
# =============================================================================


class TestBridgedBondHandling:
    """Tests for bridged bond detection and handling."""

    def test_has_bridged_bond_with_valid_molecule(self) -> None:
        """Test bridged bond detection with a simple valid molecule."""
        mol = Chem.MolFromSmiles("CCO")  # Ethanol - no bridged bonds
        assert mol is not None
        assert not has_bridged_bond(mol)

    def test_has_bridged_bond_with_none(self) -> None:
        """Test bridged bond detection with None molecule."""
        result = has_bridged_bond(None)
        # Should handle None gracefully
        assert result is True or result is False

    def test_bridged_bond_with_gt(
        self,
    ) -> None:
        """Test that bridged bond molecules are properly masked in rewards."""
        # Use known valid SMILES
        mols = [
            Chem.MolFromSmiles("N1(C(=O)C2=C(N=CN2)N(C)C1=O)CCC1CC2CC1CC2"),
            Chem.MolFromSmiles("C1=CC=C(CC2CC=C34CC(C=C3)C=C4C=2)C=C1"),
            Chem.MolFromSmiles("C1CCCCC2(CC3CC(CC3)C2)C1"),
        ]
        assert all(has_bridged_bond(mol) for mol in mols)
