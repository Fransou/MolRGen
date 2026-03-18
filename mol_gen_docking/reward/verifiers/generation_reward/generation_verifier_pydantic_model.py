import os
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from mol_gen_docking.reward.verifiers.abstract_verifier_pydantic_model import (
    VerifierOutputModel,
)


class DockingConfigModel(BaseModel):
    """Pydantic model for docking configuration.

    This model defines the configuration parameters for docking operations,
    providing validation and documentation for all docking options.

    Attributes:
        exhaustiveness: Docking exhaustiveness parameter.
        n_cpu: Number of CPUs to use for docking.
        docking_oracle: Type of docking oracle to use ("pyscreener" or "autodock_gpu").
    """

    exhaustiveness: int = Field(
        default=8,
        gt=1,
        description="Docking exhaustiveness parameter",
    )

    n_cpu: int = Field(
        default=8,
        gt=1,
        description="Number of CPUs to use for docking",
    )

    docking_oracle: Literal["pyscreener", "autodock_gpu"] = Field(
        default="autodock_gpu",
        description='Type of docking oracle: "pyscreener" or "autodock_gpu"',
    )


class PyscreenerConfigModel(DockingConfigModel):
    """Pydantic model for PyScreener docking configuration.

    This model defines the configuration parameters specific to the PyScreener
    docking software, providing validation and documentation for all options.

    Attributes:
        exhaustiveness: Docking exhaustiveness parameter.
        n_cpu: Number of CPUs to use for docking.
        docking_oracle: Type of docking oracle to use (must be "pyscreener").
        software_class: Docking software class to use with PyScreener.
    """

    software_class: Literal[
        "vina",
        "qvina",
        "smina",
        "psovina",
        "dock",
        "dock6",
        "ucsfdock",
    ] = Field(
        default="vina",
        description="Docking software class to use with PyScreener",
    )

    @model_validator(mode="after")
    def check_software_class(self) -> "PyscreenerConfigModel":
        assert self.docking_oracle == "pyscreener", (
            "software_class is only valid for pyscreener docking_oracle"
        )
        return self


class DockingGPUConfigModel(DockingConfigModel):
    """Pydantic model for AutoDock GPU docking configuration.

    This model defines the configuration parameters specific to the AutoDock GPU
    docking software, providing validation and documentation for all options.

    Attributes:
        exhaustiveness: Docking exhaustiveness parameter.
        n_cpu: Number of CPUs to use for docking.
        docking_oracle: Type of docking oracle to use (must be "autodock_gpu").
        vina_mode: Command mode for AutoDock GPU.
    """

    vina_mode: str = Field(
        default="autodock_gpu_256wi",
        description="Command mode for AutoDock GPU",
    )

    @model_validator(mode="after")
    def check_vina_mode(self) -> "DockingGPUConfigModel":
        assert self.docking_oracle == "autodock_gpu", (
            "vina_mode is only valid for autodock_gpu docking_oracle"
        )
        return self


class GenerationVerifierConfigModel(BaseModel):
    """Pydantic model for generation verifier configuration.

    This model defines the configuration parameters for the GenerationVerifier class,
    providing validation and documentation for all configuration options.

    Attributes:
        path_to_mappings: Optional path to property mappings and docking targets configuration directory.
                         Should contain 'names_mapping.json' and 'docking_targets.json' files.
        reward: Type of reward to compute. Either "property" for property-based rewards or "valid_smiles"
                for validity-based rewards.
        rescale: Whether to rescale the rewards to a normalized range.
        oracle_kwargs: Dictionary of keyword arguments to pass to the docking oracle. Can include:

                       - exhaustiveness: Docking exhaustiveness parameter
                       - n_cpu: Number of CPUs for docking
                       - docking_oracle: Type of docking oracle ("pyscreener" or "autodock_gpu")
                       - vina_mode: Command mode for AutoDock GPU
        docking_concurrency_per_gpu: Number of concurrent docking runs to allow per GPU.
                                     Default is 2 (uses ~1GB per run on 80GB GPU).
    """

    path_to_mappings: str = Field(
        description="Path to property mappings and docking targets configuration directory (must contain names_mapping.json and docking_targets.json)",
    )

    reward: Literal["property", "valid_smiles"] = Field(
        default="property",
        description='Reward type: "property" for property-based or "valid_smiles" for validity-based rewards',
    )

    rescale: bool = Field(
        default=True,
        description="Whether to rescale rewards to a normalized range",
    )

    oracle_kwargs: DockingGPUConfigModel | PyscreenerConfigModel = Field(
        default_factory=DockingGPUConfigModel,
        description="Keyword arguments for the docking oracle (exhaustiveness, n_cpu, docking_oracle, vina_mode, etc.)",
    )

    docking_concurrency_per_gpu: int = Field(
        default=2,
        gt=0,
        description="Number of concurrent docking runs per GPU (each uses ~1GB on 80GB GPU)",
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
        json_schema_extra = {
            "example": {
                "path_to_mappings": "data/molgendata",
                "reward": "property",
                "rescale": True,
                "oracle_kwargs": {
                    "exhaustiveness": 8,
                    "n_cpu": 8,
                    "docking_oracle": "autodock_gpu",
                    "vina_mode": "autodock_gpu_256wi",
                },
                "docking_concurrency_per_gpu": 2,
            }
        }

    @model_validator(mode="after")
    def check_mappings_path(self) -> "GenerationVerifierConfigModel":
        """Validate that the path_to_mappings exists and contains required files."""
        if self.path_to_mappings is not None:
            if not os.path.exists(self.path_to_mappings):
                raise ValueError(
                    f"Path to mappings {self.path_to_mappings} does not exist."
                )
            names_mapping_path = os.path.join(
                self.path_to_mappings, "names_mapping.json"
            )
            docking_targets_path = os.path.join(
                self.path_to_mappings, "docking_targets.json"
            )
            if not os.path.exists(names_mapping_path):
                raise ValueError(
                    f"names_mapping.json not found at {names_mapping_path}"
                )
            if not os.path.exists(docking_targets_path):
                raise ValueError(
                    f"docking_targets.json not found at {docking_targets_path}"
                )
        return self


class GenerationVerifierMetadataModel(BaseModel):
    """Metadata model for generation verifier results.

    Contains detailed information about the generation verification process,
    including all extracted SMILES, their individual rewards, and any extraction failures.

    Attributes:
        properties: List of property names that were evaluated (e.g., "docking_score", "QED", "SA").
            Each property corresponds to a molecular descriptor or docking target that was optimized.

        individual_rewards: List of individual rewards for each property in the properties list.
            Each value is not necessarily in [0.0, 1.0], representing
            how well the molecule satisfies each property objective.

        all_smi_rewards: List of rewards for all SMILES found in the completion.
            When multiple SMILES are extracted, each gets its own reward. The final reward
            is typically the best among these values.

        all_smi: List of all SMILES strings extracted from the completion.
            May contain multiple SMILES if the model generated several molecules.
            Empty if SMILES extraction failed.

        smiles_extraction_failure: Error message if SMILES extraction failed.
            Empty string if extraction was successful. Common values include:

            - "no_smiles": No valid SMILES found in the completion
            - "multiple_smiles_in_boxed": Multiple SMILES found when only one was expected
            - "invalid_smiles": SMILES string found but not chemically valid
    """

    properties: List[str] = Field(
        default_factory=list,
        description="List of property names that were evaluated.",
    )
    individual_rewards: List[float] = Field(
        default_factory=list,
        description="List of individual rewards for each property.",
    )
    all_smi_rewards: List[float] = Field(
        default_factory=list,
        description="List of rewards for all SMILES in the completion.",
    )
    all_smi: List[str] = Field(
        default_factory=list,
        description="List of all SMILES strings in the completion.",
    )
    smiles_extraction_failure: str = Field(
        default="",
        description="Error message if there was a failure in extracting SMILES from the completion.",
        frozen=False,
    )


class GenerationVerifierOutputModel(VerifierOutputModel):
    """Output model for generation verifier results.

    Attributes:
        reward: The computed reward for the generation verification.
        parsed_answer: The parsed answer extracted from the model completion.
        verifier_metadata: Metadata related to the generation verification process.
    """

    reward: float = Field(
        ...,
        description="The computed reward for the generation verification.",
    )
    parsed_answer: str = Field(
        ..., description="The parsed answer extracted from the model completion."
    )
    verifier_metadata: GenerationVerifierMetadataModel = Field(
        ...,
        description="Metadata related to the generation verification process.",
    )
