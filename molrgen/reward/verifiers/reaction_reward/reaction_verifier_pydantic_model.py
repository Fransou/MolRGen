import os
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from molrgen.reward.verifiers.abstract_verifier_pydantic_model import (
    VerifierOutputModel,
)

ReactionObjT = Literal[
    "final_product",
    "reactant",
    "all_reactants",
    "all_reactants_bb_ref",
    "smarts",
    "full_path",
    "full_path_bb_ref",
    "full_path_smarts_ref",
    "full_path_smarts_bb_ref",
    "analog_gen",
]


class ReactionVerifierConfigModel(BaseModel):
    """Pydantic model for reaction verifier configuration.

    This model defines the configuration parameters for the ReactionVerifierConfigModel class,
    providing validation and documentation for all configuration options related to reaction
    verification, parsing methods, reward types, and reaction matrix paths.

    Attributes:
        parsing_method: Method to parse model completions for SMILES or property values.
        reward: Reward type, either "property" for property-based or "valid_smiles" for validity-based rewards.
        reaction_matrix_path: Path to the reaction matrix pickle file for reaction verification.
        reaction_reward_type: For retro-synthesis, assign reward based on exact match (binary) or Tanimoto similarity.
        pg_name: Optional name of the Ray placement group to schedule tasks on.
    """

    parsing_method: Literal["none", "answer_tags", "boxed"] = Field(
        default="answer_tags",
        description="Method to parse model completions for SMILES or property values.",
    )

    reward: Literal["property", "valid_smiles"] = Field(
        default="property",
        description='Reward type: "property" for property-based or "valid_smiles" for validity-based rewards',
    )

    reaction_matrix_path: str = Field(
        default="data/rxn_matrix.pkl",
        description="Path to the reaction matrix pickle file for reaction verification",
    )

    reaction_reward_type: Literal["binary", "tanimoto"] = Field(
        default="tanimoto",
        description="For retro-synthesis, assign reward based on the exact match (binary) or Tanimoto similarity of the last product",
    )

    pg_name: Optional[str] = Field(
        default=None,
        description="Optional name of the Ray placement group to schedule tasks on.",
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "reward": "property",
                "reaction_matrix_path": "data/rxn_matrix.pkl",
                "reaction_reward_type": "tanimoto",
            }
        }

    @model_validator(mode="after")
    def check_reaction_matrix_path(self) -> "ReactionVerifierConfigModel":
        """Validate that the reaction matrix path exists."""
        if not os.path.exists(self.reaction_matrix_path):
            raise ValueError(
                f"Reaction matrix path {self.reaction_matrix_path} does not exist."
            )
        return self


class ReactionVerifierMetadataModel(BaseModel):
    """Metadata model for reaction verifier results.

    Contains detailed information about the reaction verification process,
    including validity, product correctness, and reactant validation.

    Attributes:
        valid: Proportion of valid reaction steps (0.0 to 1.0).
            For single reaction tasks: 1.0 if valid, 0.0 if invalid.
            For synthesis route tasks (full_path): proportion of reaction steps that are chemically valid.

        correct_product: Whether the product is correct or similarity to the target molecule.
            For SMARTS prediction tasks: 1.0 if reaction produces the correct product, 0.0 otherwise.
            For synthesis tasks with tanimoto similarity: Tanimoto similarity score (0.0 to 1.0)
            between the final product and the target molecule.
            For exact match tasks: 1.0 if exact match, 0.0 otherwise.

        correct_reactant: Whether all reactants are correct.
            For building block constrained tasks: True if all reactants are in the allowed building blocks list.
            For unconstrained tasks: True if all reactants are chemically valid.
            False if any reactant is invalid or not allowed.
    """

    valid: float = Field(
        default=0.0,
        description="Is the answer valid. If the task is to propose a synthesis route, this is the proportion of valid reaction steps (0.0 to 1.0).",
    )
    correct_product: float = Field(
        default=0.0,
        description="Whether the product is correct. For synthesis tasks, if we use tanimoto similarity, similarity to the target molecule, for SMARTS prediction, do both of the chemical reactions lead to the correct product.",
    )
    correct_reactant: bool = Field(
        default=False,
        description="Whether all reactants are correct.",
    )


class ReactionVerifierOutputModel(VerifierOutputModel):
    """Output model for reaction verifier results.

    Attributes:
        reward: The computed reward for the reaction verification.
        parsed_answer: The parsed answer extracted from the model completion.
        verifier_metadata: Metadata related to the reaction verification process.
    """

    reward: float = Field(
        ...,
        description="The computed reward for the reaction verification.",
    )
    parsed_answer: str = Field(
        ..., description="The parsed answer extracted from the model completion."
    )
    verifier_metadata: ReactionVerifierMetadataModel = Field(
        ...,
        description="Metadata related to the reaction verification process.",
    )
