from typing import Literal, Optional

from pydantic import BaseModel, Field

from molrgen.reward.verifiers.abstract_verifier_pydantic_model import (
    VerifierOutputModel,
)


class MolPropVerifierConfigModel(BaseModel):
    """Pydantic model for molecular verifier configuration.

    This model defines the configuration parameters for the MolecularVerifier class,
    providing validation and documentation for all configuration options.
    """

    reward: Literal["property", "valid_smiles"] = Field(
        default="property",
        description='Reward type: "property" for property-based or "valid_smiles" for validity-based rewards',
    )
    parsing_method: Literal["none", "answer_tags", "boxed"] = Field(
        default="answer_tags",
        description="Method to parse model completions for SMILES or property values.",
    )

    pg_name: Optional[str] = Field(
        default=None,
        description="Optional name of the Ray placement group to schedule tasks on.",
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        validate_assignment = True
        json_schema_extra = {
            "example": {
                "reward": "property",
                "parsing_method": "answer_tags",
            }
        }


class MolPropVerifierMetadataModel(BaseModel):
    """Metadata model for molecular property verifier results.

    Contains detailed information about the property prediction verification process,
    including the extracted answer and whether extraction was successful.

    Attributes:
        extracted_answer: The numerical answer extracted from the model completion.
            For regression tasks, this is the predicted property value.
            For classification tasks, this is the predicted class (0 or 1).

        extraction_success: Indicates whether the answer extraction was successful.
            False if the answer could not be parsed from the completion text,
            or if the answer format was invalid (e.g., non-numeric for regression).
    """

    extracted_answer: float = Field(
        ...,
        description="The extracted answer string from the model completion.",
    )
    extraction_success: bool = Field(
        ...,
        description="Indicates whether the answer extraction was successful.",
    )


class MolPropVerifierOutputModel(VerifierOutputModel):
    """Output model for molecular property verifier results.

    Attributes:
        reward: The computed reward for the molecular property verification.
        parsed_answer: The parsed answer extracted from the model completion.
        verifier_metadata: Metadata related to the molecular property verification process.
    """

    reward: float = Field(
        ...,
        description="The computed reward for the molecular property verification.",
    )
    parsed_answer: str = Field(
        ...,
        description="The parsed answer from the model completion.",
    )
    verifier_metadata: MolPropVerifierMetadataModel = Field(
        ...,
        description="Metadata related to the molecular property verification process.",
    )
