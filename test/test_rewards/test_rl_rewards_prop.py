"""Tests for property-based reward scoring."""

from typing import Any, List

import numpy as np
import pytest

from mol_gen_docking.reward import (
    GenerationVerifierConfigModel,
    MolecularVerifier,
    MolecularVerifierConfigModel,
    MolPropVerifierConfigModel,
)
from mol_gen_docking.utils.property_utils import rescale_property_values


@pytest.fixture(scope="module", params=["answer_tags", "boxed"])  # type: ignore
def property_scorer(
    data_path: str, request: pytest.FixtureRequest
) -> MolecularVerifier:
    """Create a RewardScorer for property scoring."""
    return MolecularVerifier(
        MolecularVerifierConfigModel(
            parsing_method=request.param,
            mol_prop_verifier_config=MolPropVerifierConfigModel(),
        )
    )


@pytest.fixture(scope="module")  # type: ignore
def property_scorer_mixed(data_path: str) -> MolecularVerifier:
    """Create a RewardScorer for property scoring."""
    return MolecularVerifier(
        MolecularVerifierConfigModel(
            mol_prop_verifier_config=MolPropVerifierConfigModel(),
            generation_verifier_config=GenerationVerifierConfigModel(
                path_to_mappings=data_path
            ),
        )
    )


@pytest.fixture(scope="module", params=[0.01, 0.1, 1, 10, 100])  # type: ignore
def scale(request: pytest.FixtureRequest) -> int:
    """Scale factor for regression tests."""
    out: int = request.param
    return out


# =============================================================================
# Regression Tests
# =============================================================================


class TestRegression:
    """Tests for regression objective."""

    def create_scientific_notation(self, value: float) -> list[str]:
        return [
            f"{value / 10**exp:f}{notation}{exp}"
            for exp in [-2, -1, 0, 1, 2]
            for notation in ["e", "E"]
        ]

    def create_latex_notation(self, value: float) -> list[str]:
        return [
            rf"{value / 10**exp:f} {mult} 10^{exp}".format(value / (10**exp), exp)
            for exp in [-2, -1, 0, 1, 2]
            for mult in [r"\times", "×"]
        ]

    def create_html_notation(self, value: float) -> list[str]:
        return [
            f"{value / 10**exp:f} x 10<sup>{exp}</sup>".format(value / (10**exp), exp)
            for exp in [-2, -1, 0, 1, 2]
        ]

    def create_pm_notation(self, value: float) -> list[str]:
        return [
            f"{value} {pm} {uncertainty}"
            for uncertainty in [0.1 * value, 0.2 * value]
            for pm in ["±", "+/-", "+-"]
        ]

    def create_range_notation(self, value: float) -> list[str]:
        return [f"{value - 0.1 * value} - {value + 0.1 * value}"]

    @pytest.mark.parametrize("style", ["scientific", "latex", "html", "pm", "range"])  # type: ignore
    def test_regression(
        self, property_scorer: MolecularVerifier, scale: int, style: str
    ) -> None:
        """Test that regression rewards are properly ordered by value."""
        target: float = np.random.random() / scale
        values = [
            v * np.sign(np.random.random() - 0.5) + target for v in [0, 0.01, 0.5, 1]
        ]
        if style == "scientific":
            val_strs = sum([self.create_scientific_notation(v) for v in values], [])
            n_repets = 10
        elif style == "latex":
            val_strs = sum([self.create_latex_notation(v) for v in values], [])
            n_repets = 10
        elif style == "html":
            val_strs = sum([self.create_html_notation(v) for v in values], [])
            n_repets = 5
        elif style == "pm":
            val_strs = sum([self.create_pm_notation(v) for v in values], [])
            n_repets = 6
        elif style == "range":
            val_strs = sum([self.create_range_notation(v) for v in values], [])
            n_repets = 1
        else:
            raise ValueError(f"unknown style: {style}")
        if property_scorer.verifier_config.parsing_method == "answer_tags":
            completions: List[str] = [
                "<answer> Here is an answer: {} </answer>".format(v) for v in val_strs
            ]
        elif property_scorer.verifier_config.parsing_method == "boxed":

            def get_corr_comp(v: str) -> str:
                if np.random.random() > 0.5:
                    return "Here is an answer: 128.3 \\boxed{{ {} }}".format(v)
                else:
                    return "<answer>Here is an answer: 128.3 \\boxed{{ {} }}</answer>".format(
                        v
                    )

            completions = ["\\boxed{123111315}" + get_corr_comp(v) for v in val_strs]
        else:
            raise ValueError(
                f"unknown parsing method: {property_scorer.verifier_config.parsing_method}"
            )
        metadata: List[dict[str, Any]] = [
            {"objectives": ["regression"], "properties": [""], "target": [target]}
        ] * len(completions)

        verif_metadata: Any = property_scorer(completions, metadata).verifier_metadatas
        assert verif_metadata is not None
        extracted_answers = [
            [
                m.mol_prop_verifier_metadata.extracted_answer
                for m in verif_metadata[i : i + n_repets]
            ]
            for i in range(0, len(verif_metadata), n_repets)
        ]
        for ext_ans, expected_val in zip(extracted_answers, values):
            assert all(np.isclose(ext, expected_val, atol=1e-3) for ext in ext_ans), (
                f"extracted: {ext_ans}, expected: {expected_val}"
            )


# =============================================================================
# Classification Tests
# =============================================================================


class TestClassification:
    """Tests for classification objective."""

    def test_classification_target_one(
        self, property_scorer: MolecularVerifier
    ) -> None:
        """Test classification with target=1."""
        values = [1, 1, 0, 0, "bbfhdsbfsj", "Y."]
        if property_scorer.verifier_config.parsing_method == "answer_tags":
            completions: List[str] = [
                "<answer> My answer is {} </answer>".format(v) for v in values
            ]
        elif property_scorer.verifier_config.parsing_method == "boxed":
            completions = [
                "My answer is \\boxed{{64}} <answer> \\boxed{{ {} }} </answer>".format(
                    v
                )
                for v in values
            ]
        else:
            raise ValueError(
                f"unknown parsing method: {property_scorer.verifier_config.parsing_method}"
            )
        metadata: List[dict[str, Any]] = [
            {"objectives": ["classification"], "properties": [""], "target": [1]}
        ] * 6
        rewards: List[float] = property_scorer(completions, metadata).rewards
        assert rewards == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    def test_classification_target_zero(
        self, property_scorer: MolecularVerifier
    ) -> None:
        """Test classification with target=0."""
        values = [1, 1, 0, 0, "bbfhdsbfsj"]
        if property_scorer.verifier_config.parsing_method == "answer_tags":
            completions: List[str] = [
                "<answer> My answer is {} </answer>".format(v) for v in values
            ]
        elif property_scorer.verifier_config.parsing_method == "boxed":
            completions = [
                "<answer> My answer is \\boxed{{ {} }}</answer>".format(v)
                for v in values
            ]
        else:
            raise ValueError(
                f"unknown parsing method: {property_scorer.verifier_config.parsing_method}"
            )
        metadata: List[dict[str, Any]] = [
            {"objectives": ["classification"], "properties": [""], "target": [0]}
        ] * 5
        rewards: List[float] = property_scorer(completions, metadata).rewards
        assert rewards == [0.0, 0.0, 1.0, 1.0, 0.0]


# =============================================================================
# Mixed Task Tests
# =============================================================================


class TestMixedGeneration:
    """Tests for mixed classification and generation tasks."""

    def test_with_generation(self, property_scorer_mixed: MolecularVerifier) -> None:
        """Test mixed classification and generation with target=1."""
        completions: List[str] = sum(
            [
                [
                    "<answer> \\boxed{{ {} }} </answer>".format(v),
                    "<answer> \\boxed{{ CCC{} }} </answer>".format(smi),
                ]
                for v, smi in zip([1, 1, 0, 0, 0], ["C" * i for i in range(5)])
            ],
            [],
        )
        metadata: List[dict[str, Any]] = sum(
            [
                [
                    {
                        "objectives": ["classification"],
                        "properties": [""],
                        "target": [1],
                    },
                    {
                        "objectives": ["maximize"],
                        "properties": ["CalcNumRotatableBonds"],
                        "target": [1],
                    },
                ]
            ]
            * 5,
            [],
        )
        rewards = property_scorer_mixed(completions, metadata, debug=True).rewards
        assert rewards[::2] == [1.0, 1.0, 0.0, 0.0, 0.0]
        assert (
            rewards[1::2]
            == np.array(
                [rescale_property_values("CalcNumRotatableBonds", i) for i in range(5)]
            ).clip(0, 1)
        ).all()

    def test_with_generation_no_gen_conf(
        self, property_scorer: MolecularVerifier
    ) -> None:
        """Test mixed classification and generation with target=0."""
        values = [1, 1, 0, 0, 0]
        smis = ["C" * i for i in range(5)]
        if property_scorer.verifier_config.parsing_method == "answer_tags":
            completions: List[str] = [
                "<answer> {} </answer>".format(v) for v in values
            ] + ["<answer> CCC{} </answer>".format(smi) for smi in smis]
        elif property_scorer.verifier_config.parsing_method == "boxed":
            completions = [
                "<answer> \\boxed{{ {} }} </answer>".format(v) for v in values
            ] + [
                "<answer> \\boxed{{CCC{}}} </answer>".format(
                    smi,
                )
                for smi in smis
            ]
        else:
            raise ValueError(
                f"unknown parsing method: {property_scorer.verifier_config.parsing_method}"
            )
        metadata: List[dict[str, Any]] = [
            {"objectives": ["classification"], "properties": [""], "target": [0]}
        ] * 5 + [
            {
                "objectives": ["maximize"],
                "properties": ["CalcNumRotatableBonds"],
                "target": [1],
            }
        ] * 5
        with pytest.raises(AssertionError):
            property_scorer(completions, metadata, debug=True)
