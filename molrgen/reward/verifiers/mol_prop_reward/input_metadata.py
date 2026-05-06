from typing import List, Literal

from pydantic import BaseModel, Field

MolPropObjT = Literal["regression", "classification"]


class MolPropVerifierInputMetadataModel(BaseModel):
    """Input metadata model for molecular property verifier.

    Defines the verification criteria for molecular property prediction tasks,
    including the type of objective (regression or classification), properties
    being predicted, target values, and normalization parameters.

    Attributes:
        objectives: List of objective types for each property.
            Valid values:
            - "regression": Continuous value prediction (e.g., predicting logP, molecular weight)
            - "classification": Binary classification (0 or 1)
            Must have the same length as properties and target.

        properties: List of molecular property names being predicted.
            Optional, used for tracking and logging purposes.
            Examples: "logP", "MW", "toxicity", "solubility"

        target: List of target values for each property to verify against.
            For regression: The ground truth continuous value
            For classification: The ground truth class (0 or 1)
            Must have the same length as objectives.

        norm_var: Normalization variance for regression objectives.
            Optional parameter used to compute normalized rewards for regression tasks.
            The reward is computed as: reward = max(0, 1 - |predicted - target| / norm_var)
            If None, no normalization is applied and absolute error is used directly.
    """

    objectives: List[MolPropObjT] = Field(
        ...,
        description="The type of objective for the property: regression or classification.",
    )
    properties: List[str] = Field(
        default_factory=list,
        description="The molecular properties to be verified.",
    )
    target: List[float | int] = Field(
        ...,
        description="The target value for the molecular property to verify against.",
    )
    norm_var: float | int | None = Field(
        default=None,
        description="Normalization variance for regression objectives.",
    )
