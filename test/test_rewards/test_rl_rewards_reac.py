"""Tests for reaction-based reward scoring."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from molrgen.data.pydantic_dataset import read_jsonl
from molrgen.reward import (
    MolecularVerifier,
    MolecularVerifierConfigModel,
    ReactionVerifierConfigModel,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")  # type: ignore
def property_scorer(data_path: str) -> MolecularVerifier:
    """Create a RewardScorer for property scoring without rescaling."""
    return MolecularVerifier(
        MolecularVerifierConfigModel(
            reward="property", reaction_verifier_config=ReactionVerifierConfigModel()
        )
    )


@pytest.fixture(scope="module")  # type: ignore
def property_scorer_valid(data_path: str) -> MolecularVerifier:
    """Create a RewardScorer for valid SMILES scoring without rescaling."""
    return MolecularVerifier(
        MolecularVerifierConfigModel(
            reward="valid_smiles",
            reaction_verifier_config=ReactionVerifierConfigModel(),
        )
    )


@pytest.fixture(scope="module")  # type: ignore
def dataset_reac() -> List[Any]:
    """Load the reaction test dataset."""
    current_path = os.path.dirname(os.path.abspath(__file__))
    return read_jsonl(
        Path(
            os.path.join(os.path.dirname(current_path), "data", "reactions_test.jsonl")
        )
    )


@pytest.fixture(scope="module")  # type: ignore
def add_sy_ex() -> List[Dict[str, Any]]:
    """Load the additional synthesis full path examples."""
    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(
            os.path.dirname(current_path),
            "data",
            "reaction_full_sythn_test_examples.json",
        )
    ) as f:
        examples: List[Dict[str, Any]] = json.load(f)
    return examples


@pytest.fixture(scope="module", params=list(range(100)))  # type: ignore
def reaction_dataset_line(
    request: pytest.FixtureRequest, dataset_reac: List[Any]
) -> Any:
    return dataset_reac[request.param]


# =============================================================================
# Reaction Tests
# =============================================================================


class TestReaction:
    """Tests for reaction reward scoring."""

    def test_reaction(
        self,
        reaction_dataset_line: Any,
        property_scorer: MolecularVerifier,
        property_scorer_valid: MolecularVerifier,
        properties_csv: pd.DataFrame,
    ) -> None:
        """Test reaction reward scoring for different objective types."""
        metadata = reaction_dataset_line.conversations[0].meta
        target = metadata["target"]
        if metadata["objectives"][0].startswith("full_path"):
            reactants = metadata["reactants"]
            products = metadata["products"]
            real: dict[str, Any] = {"answer": []}
            for i, (r, p) in enumerate(zip(reactants, products)):
                real["answer"].append({"step": i + 1, "reactants": r, "product": [p]})

            fake: dict[str, Any] = {"answer": []}
            for i, (r, p) in enumerate(zip(reactants, products)):
                fake["answer"].append(
                    {
                        "step": i + 1,
                        "reactants": r,
                        "product": [properties_csv.smiles.sample(1).values[0]],
                    }
                )

            fake2: dict[str, Any] = {"answer": []}
            for i, (r, p) in enumerate(zip(reactants, products)):
                fake2["answer"].append(
                    {
                        "step": i + 1,
                        "reactants": [0.5] * len(r),
                        "product": [properties_csv.smiles.sample(1).values[0]],
                    }
                )
            fake3: dict[str, Any] = {"answer": []}
            for i, (r, p) in enumerate(zip(reactants, products)):
                fake3["answer"].append(
                    {
                        "step": i + 1,
                        "product": [properties_csv.smiles.sample(1).values[0]],
                    }
                )

            answers = [
                json.dumps(real),
                json.dumps(fake),
                json.dumps(fake2),
                json.dumps(fake3),
            ]

        elif metadata["objectives"][0] in [
            "final_product",
            "reactant",
        ]:
            answers = []
            for pred_sol in [target] + [
                properties_csv.smiles.sample(1).tolist() for i in range(1, 3)
            ]:
                np.random.shuffle(pred_sol)
                answers.append(json.dumps({"answer": pred_sol[0]}))
        elif metadata["objectives"][0] in [
            "all_reactants",
            "all_reactants_bb_ref",
        ]:
            answers = []
            for pred_sol in [target] + [
                properties_csv.smiles.sample(i).tolist() for i in range(1, 3)
            ]:
                answers.append(json.dumps({"answer": pred_sol}))
        elif metadata["objectives"][0] in ["smarts"]:
            fakes = [
                property_scorer.reaction_verifier.rxn_matrix._reactions[0].smarts,
                property_scorer.reaction_verifier.rxn_matrix._reactions[10].smarts,
                "[#6:1]-[N:5]=[N+:6]=[N-:7].[#6:2]-[C:3]>>[#6:2][cH0+0:3]1[cH1+0:4][nH0+0:5][nH0+0:6][nH0+0:7]1",
            ]
            answers = []
            for pred_sol in [target[0]] + fakes:
                answers.append(json.dumps({"answer": pred_sol}))

        answers.append("{npkdf;lds} jij}")
        completions = ["<answer>\n {} \n</answer>".format(v) for v in answers]

        result = property_scorer(completions, [metadata] * len(answers))
        rewards = result.rewards
        assert rewards[0] == 1
        assert all(r == 0 for r in rewards[1:])


# =============================================================================
# Additional Synthesis Tests
# =============================================================================


class TestAdditionalSynthesis:
    """Tests for additional synthesis full path examples."""

    def test_additional_synth_full_path(
        self,
        add_sy_ex: List[Dict[str, Any]],
        property_scorer: MolecularVerifier,
        property_scorer_valid: MolecularVerifier,
    ) -> None:
        """Test additional synthesis examples."""
        for in_out in add_sy_ex:
            metadata = in_out["metadata"]
            completions = [in_out["output"]]
            property_scorer(completions, [metadata]).rewards
