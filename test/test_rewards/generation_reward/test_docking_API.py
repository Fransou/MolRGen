import os
from typing import Any, Callable, Dict, List, Union

import numpy as np
import pytest
import torch

from mol_gen_docking.data.meeko_process import ReceptorProcess
from mol_gen_docking.reward import (
    GenerationVerifierConfigModel,
    MolecularVerifier,
    MolecularVerifierConfigModel,
)


@pytest.fixture(scope="module", params=[4, 8, 16])  # type:ignore
def exhaustiveness(request: pytest.FixtureRequest) -> int:
    """Fixture for testing different exhaustiveness levels."""
    out: int = request.param
    return out


@pytest.fixture(scope="module")  # type:ignore # type:ignore
def scorer(has_gpu: bool, data_path: str, exhaustiveness: int) -> MolecularVerifier:
    """Create a RewardScorer configured for docking tests."""
    if not has_gpu:
        return MolecularVerifier(
            MolecularVerifierConfigModel(
                reward="property",
                parsing_method="none",
                generation_verifier_config=GenerationVerifierConfigModel(
                    rescale=False,
                    path_to_mappings=data_path,
                    oracle_kwargs=dict(
                        n_cpu=int(os.environ.get("N_CPUS_DOCKING", exhaustiveness)),
                        exhaustiveness=exhaustiveness,
                        docking_oracle="pyscreener",
                    ),
                ),
            )
        )

    else:
        return MolecularVerifier(
            MolecularVerifierConfigModel(
                reward="property",
                parsing_method="none",
                generation_verifier_config=GenerationVerifierConfigModel(
                    rescale=False,
                    path_to_mappings=data_path,
                    oracle_kwargs=dict(
                        n_cpu=int(os.environ.get("N_CPUS_DOCKING", exhaustiveness)),
                        exhaustiveness=exhaustiveness,
                        docking_oracle="autodock_gpu",
                        docking_executable="autodock_gpu_128wi",
                    ),
                ),
            )
        )


@pytest.fixture(scope="module")  # type:ignore
def receptor_process(
    has_gpu: bool, data_path: str
) -> Callable[[Union[str, List[str]]], tuple[List[str], List[str]]]:
    """Create a ReceptorProcess for GPU or return a dummy function for CPU."""
    if not has_gpu:
        return lambda x: ([], [])
    else:
        rp = ReceptorProcess(data_path=data_path, pre_process_receptors=True)

        def wrapped_fn(r: Union[str, List[str]]) -> tuple[List[str], List[str]]:
            if isinstance(r, str):
                r = [r]
            return rp.process_receptors(r, allow_bad_res=True)

        return wrapped_fn


def build_metadatas(
    property: Union[str, List[str]], obj: str = "maximize"
) -> Dict[str, Union[List[str], List[float]]]:
    """Build metadata dictionary for reward scoring.

    Args:
        property: Single property name or list of property names.
        obj: Objective type (default: "maximize").

    Returns:
        Dictionary with properties, objectives, and target values.
    """
    if isinstance(property, str):
        properties = [property]
    else:
        properties = property
    return {
        "properties": properties,
        "objectives": [obj] * len(properties),
        "target": [0.0] * len(properties),
    }


# =============================================================================
# Receptor Processing Tests
# =============================================================================


class TestReceptorProcessing:
    """Tests for receptor processing."""

    def test_receptor_process(
        self, receptor_process: Callable, docking_targets: List[str]
    ) -> None:
        """Test receptor processing for all docking targets."""
        _, missed_targets = receptor_process(docking_targets)
        assert len(missed_targets) == 0, (
            f"Receptors {missed_targets} could not be processed."
        )


# =============================================================================
# Single Property Docking Tests
# =============================================================================


class TestSinglePropertyDocking:
    """Tests for docking with single properties."""

    def test_docking_props(
        self,
        docking_targets: List[str],
        scorer: MolecularVerifier,
        receptor_process: Callable,
        properties_csv: Any,
        n_generations: int = 4,
    ) -> None:
        """Test the reward function runs for docking targets.

        Args:
            docking_targets: List of docking target names.
            scorer: MolecularVerifier instance for scoring.
            receptor_process: Function to process receptors.
            properties_csv: DataFrame containing molecular properties.
            n_generations: Number of generations to test.
        """
        for target_d in docking_targets:
            target = [target_d, "CalcPhi"]
            metadatas: List[Dict[str, Union[List[str], List[float]]]] = [
                build_metadatas(target)
            ] * n_generations
            smiles: List[List[str]] = [
                [properties_csv.iloc[i]["smiles"]] for i in range(n_generations)
            ]
            completions: List[str] = [
                f"Here is a molecule: <answer> {s} </answer> what are its properties?"
                for s in smiles
            ]
            rewards: List[float] = scorer(completions, metadatas).rewards
            assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
            rewards_tensor: torch.Tensor = torch.Tensor(rewards)
            assert not rewards_tensor.isnan().any()
            assert rewards_tensor.shape[0] == n_generations


# =============================================================================
# Multiple Property Docking Tests
# =============================================================================


class TestMultiplePropertyDocking:
    """Tests for docking with multiple properties."""

    def test_multi_docking_props(
        self,
        docking_targets: List[str],
        receptor_process: Callable,
        scorer: MolecularVerifier,
        properties_csv: Any,
        n_generations: int = 2,
    ) -> None:
        """Test the reward function runs for multiple docking targets.

        Args:
            docking_targets: List of docking target names.
            receptor_process: Function to process receptors.
            scorer: MolecularVerifier instance for scoring.
            properties_csv: DataFrame containing molecular properties.
            n_generations: Number of generations to test.
        """
        for targets in docking_targets[::2]:
            _, missed = receptor_process(targets)
            assert len(missed) == 0, f"Receptor {targets} could not be processed."
            metadatas: List[Dict[str, Union[List[str], List[float]]]] = [
                build_metadatas(targets)
            ] * n_generations
            smiles: List[List[str]] = [
                [properties_csv.iloc[i]["smiles"]] for i in range(n_generations)
            ]
            completions: List[str] = [
                f"Here is a molecule: <answer> {s} </answer> what are its properties?"
                for s in smiles
            ]
            rewards: List[float] = scorer(completions, metadatas).rewards
            assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
            rewards_tensor: torch.Tensor = torch.Tensor(rewards)
            assert not rewards_tensor.isnan().any()
            assert rewards_tensor.shape[0] == n_generations
