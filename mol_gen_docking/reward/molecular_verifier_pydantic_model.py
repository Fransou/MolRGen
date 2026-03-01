from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from mol_gen_docking.reward.verifiers import (
    GenerationVerifierConfigModel,
    GenerationVerifierMetadataModel,
    MolPropVerifierConfigModel,
    MolPropVerifierMetadataModel,
    ReactionVerifierConfigModel,
    ReactionVerifierMetadataModel,
)


class MolecularVerifierConfigModel(BaseModel):
    """Pydantic model for molecular verifier configuration.

    This model defines the configuration parameters for the MolecularVerifier class,
    providing validation and documentation for all configuration options.

    The reward field is automatically propagated to all sub-verifier configurations,
    ensuring consistent reward type usage across all verifiers.

    Attributes:
        parsing_method: Method to parse model completions:

            - "none": No parsing, use full completion (not recommended, risks of ambiguity in the answer extraction)
            - "answer_tags": Extract content within <answer>...</answer> tags
            - "boxed": Extract content within answer tags and boxed in the '\boxed{...}' LaTeX command
        reward: Type of reward to compute. Affects all sub-verifiers:

            - "property": Computes property-based rewards (docking scores, molecular properties)
            - "valid_smiles": Computes validity rewards (1.0 for valid single SMILES, 0.0 otherwise)
        pg_name: Optional name of the Ray placement group to schedule tasks on.
        generation_verifier_config: Configuration for de novo molecular generation tasks.
            Required if handling generation tasks. Contains settings for molecular property
            optimization, docking oracle configuration, and SMILES extraction.
        mol_prop_verifier_config: Configuration for molecular property prediction tasks.
            Required if handling property prediction tasks. Contains settings for regression
            and classification objective handling.
        reaction_verifier_config: Configuration for chemical reaction and retro-synthesis tasks.
            Required if handling reaction tasks. Contains settings for synthesis path validation
            and reaction SMARTS matching.

    Notes:
        - At least one sub-verifier configuration should be provided for the verifier to work
        - The propagate_reward_to_subconfigs validator automatically sets the reward type
          and parsing_method flag in all sub-verifier configurations
        - Invalid configurations will raise Pydantic ValidationError with detailed messages

    Example:
        ```python
        from mol_gen_docking.reward import MolecularVerifierConfigModel
        from mol_gen_docking.reward.verifiers import (
            GenerationVerifierConfigModel,
            MolPropVerifierConfigModel,
            ReactionVerifierConfigModel
        )

        # Minimal configuration with generation verifier
        config = MolecularVerifierConfigModel(
            generation_verifier_config=GenerationVerifierConfigModel(
                path_to_mappings="data/molgendata"
            )
        )

        # Full configuration with all verifiers
        config = MolecularVerifierConfigModel(
            parsing_method="answer_tags",
            reward="property",
            generation_verifier_config=GenerationVerifierConfigModel(
                path_to_mappings="data/molgendata",
                rescale=True,
                oracle_kwargs={
                    "exhaustiveness": 8,
                    "n_cpu": 8,
                    "docking_oracle": "autodock_gpu",
                    "vina_mode": "autodock_gpu_256wi"
                },
                docking_concurrency_per_gpu=2
            ),
            mol_prop_verifier_config=MolPropVerifierConfigModel(
                reward="property"
            ),
            reaction_verifier_config=ReactionVerifierConfigModel(
                reaction_matrix_path="data/rxn_matrix.pkl",
                reaction_reward_type="tanimoto"
            )
        )
        ```

    See Also:
        - GenerationVerifierConfigModel: Configuration for generation tasks
        - MolPropVerifierConfigModel: Configuration for property prediction tasks
        - ReactionVerifierConfigModel: Configuration for reaction tasks
        - MolecularVerifier: Main orchestrator class using this config
    """

    parsing_method: Literal["none", "answer_tags", "boxed"] = Field(
        default="boxed",
        description="Method to parse model completions for SMILES or property values.",
    )
    reward: Literal["valid_smiles", "property"] = Field(
        default="property",
        description="Type of reward to use for molecular verification.",
    )
    pg_name: Optional[str] = Field(
        default=None,
        description="Optional name of the Ray placement group to schedule tasks on.",
    )
    generation_verifier_config: Optional[GenerationVerifierConfigModel] = Field(
        None,
        description="Configuration for generation verifier, required if reward is 'valid_smiles'.",
    )
    mol_prop_verifier_config: Optional[MolPropVerifierConfigModel] = Field(
        None,
        description="Configuration for molecular property verifier, required if reward is 'property'.",
    )
    reaction_verifier_config: Optional[ReactionVerifierConfigModel] = Field(
        None,
        description="Configuration for reaction verifier, required if reward is 'reaction'.",
    )

    @model_validator(mode="after")
    def propagate_fields_to_subconfigs(self) -> "MolecularVerifierConfigModel":
        """Propagate the fields shared to all sub-verifier configurations."""
        if self.generation_verifier_config is not None:
            self.generation_verifier_config.reward = self.reward
            self.generation_verifier_config.parsing_method = self.parsing_method
            self.generation_verifier_config.pg_name = self.pg_name
        if self.mol_prop_verifier_config is not None:
            self.mol_prop_verifier_config.parsing_method = self.parsing_method
            self.mol_prop_verifier_config.reward = self.reward
            self.mol_prop_verifier_config.pg_name = self.pg_name
        if self.reaction_verifier_config is not None:
            ### FOR REACTION VERIFIER, SPECIAL BEHAVIOR:
            ###     - if parsing_method is "none", set it to "none" for reaction verifier
            ###     - else, set it to "answer_tags" for reaction verifier (boxed not supported)
            if self.parsing_method == "none":
                self.reaction_verifier_config.parsing_method = "none"
            else:
                self.reaction_verifier_config.parsing_method = "answer_tags"
            self.reaction_verifier_config.reward = self.reward
            self.reaction_verifier_config.pg_name = self.pg_name
        return self

    class Config:
        """Pydantic configuration for the MolecularVerifierConfigModel."""

        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "parsing_method": "answer_tags",
                "reward": "property",
                "generation_verifier_config": {
                    "path_to_mappings": "data/molgendata",
                    "rescale": True,
                    "oracle_kwargs": {
                        "exhaustiveness": 8,
                        "n_cpu": 8,
                        "docking_oracle": "autodock_gpu",
                        "vina_mode": "autodock_gpu_256wi",
                    },
                    "docking_concurrency_per_gpu": 2,
                },
                "mol_prop_verifier_config": {},
                "reaction_verifier_config": {
                    "reaction_matrix_path": "data/rxn_matrix.pkl",
                    "reaction_reward_type": "tanimoto",
                },
            }
        }


class MolecularVerifierOutputMetadataModel(BaseModel):
    """Metadata model for molecular verifier output.

    This model aggregates metadata from all sub-verifiers (Generation, Molecular Property,
    and Reaction) into a unified structure. Each field may be populated or None depending
    on which verifier was active for the given task.

    Attributes:
        generation_verifier_metadata: Metadata from the generation verifier containing:
            - properties: List of evaluated property names
            - individual_rewards: Individual reward for each property
            - all_smi_rewards: Rewards for all SMILES found in completion
            - all_smi: List of all extracted SMILES strings
            - smiles_extraction_failure: Error message if SMILES extraction failed

        mol_prop_verifier_metadata: Metadata from the molecular property verifier containing:
            - extracted_answer: The numerical answer extracted from the completion
            - extraction_success: Whether the answer extraction was successful

        reaction_verifier_metadata: Metadata from the reaction verifier containing:
            - valid: Proportion of valid reaction steps (0.0 to 1.0)
            - correct_product: Product correctness or similarity to target molecule
            - correct_reactant: Whether all building blocks used are valid

    See Also:
        - GenerationVerifierMetadataModel: Metadata for generation tasks
        - MolPropVerifierMetadataModel: Metadata for property prediction tasks
        - ReactionVerifierMetadataModel: Metadata for reaction tasks
    """

    generation_verifier_metadata: Optional[GenerationVerifierMetadataModel] = Field(
        None,
        description="Metadata from the generation verifier, if applicable.",
    )
    mol_prop_verifier_metadata: Optional[MolPropVerifierMetadataModel] = Field(
        None,
        description="Metadata from the molecular property verifier, if applicable.",
    )
    reaction_verifier_metadata: Optional[ReactionVerifierMetadataModel] = Field(
        None,
        description="Metadata from the reaction verifier, if applicable.",
    )


class BatchMolecularVerifierOutputModel(BaseModel):
    """Output model for molecular verifier batch results.

    This model encapsulates the results from batch verification across multiple tasks
    and completions. It provides both the computed rewards and detailed metadata about
    each verification process, enabling comprehensive analysis of model performance.

    The output is structured as parallel lists where each index corresponds to a single
    completion from the input batch, allowing easy correlation between inputs and outputs.

    Attributes:
        rewards: List of computed reward scores, one per input completion.
            Each reward is a float value typically in the range [0.0, 1.0], though values
            may exceed 1.0 for maximize objectives depending on the input values.
            The reward value depends on the task type and objective:

            - Generation tasks: Geometric mean of per-property rewards
            - Property prediction: 0-1 for classification, 0-1 (clipped) for regression
            - Reaction tasks: 0-1 for ground truth, 0-1 for path validation
        parsed_answer: The parsed answer extracted from the model completion.
        verifier_metadatas: List of metadata objects from each verification process.
            Each element corresponds to the reward at the same index. The metadata type
            depends on which verifier processed the completion:

            - GenerationVerifierMetadataModel: For generation tasks (SMILES extraction, properties)
            - MolPropVerifierMetadataModel: For property prediction (extracted values)
            - ReactionVerifierMetadataModel: For reaction tasks (validation results)

    Notes:
        - The length of rewards and verifier_metadatas must always match the number
          of input completions
        - The order of outputs corresponds to the order of inputs
        - Metadata provides detailed diagnostic information about failures (extraction errors,
          invalid molecules, validation failures, etc.)
        - All numeric reward values are finite (not NaN or inf)

    Example:
        ```python
        from mol_gen_docking.reward import MolecularVerifier, MolecularVerifierConfigModel
        from mol_gen_docking.reward.verifiers import GenerationVerifierConfigModel

        config = MolecularVerifierConfigModel(
            generation_verifier_config=GenerationVerifierConfigModel(
                path_to_mappings="data/molgendata"
            )
        )
        verifier = MolecularVerifier(config)

        # Verify multiple completions
        completions = [
            "<answer>CCO</answer>",
            "<answer>c1ccccc1</answer>",
            "<answer>invalid_smiles</answer>"
        ]
        metadata = [
            {"properties": ["QED"], "objectives": ["maximize"], "target": [0.0]},
            {"properties": ["QED"], "objectives": ["maximize"], "target": [0.0]},
            {"properties": ["QED"], "objectives": ["maximize"], "target": [0.0]}
        ]

        results = verifier(completions, metadata)

        # Access results
        print(f"Number of completions: {len(results.rewards)}")
        for i, (reward, meta) in enumerate(zip(results.rewards, results.verifier_metadatas)):
            print(f"Completion {i}: reward={reward:.3f}")
            print(f"  Properties: {meta.properties}")
            print(f"  All SMILES: {meta.all_smi}")
            if meta.smiles_extraction_failure:
                print(f"  Failure reason: {meta.smiles_extraction_failure}")

        #  Output:
        # >>> Number of completions: 3
        # Completion 0: reward=0.113
        #   Properties: ['QED']
        #   All SMILES: ['CCO']
        # Completion 1: reward=0.173
        #   Properties: ['QED']
        #   All SMILES: ['c1ccccc1']
        # Completion 2: reward=0.000
        #   Properties: []
        #   All SMILES: []
        #   Failure reason: no_smiles
        ```


    See Also:
        - MolecularVerifier: Main class that returns this model
        - GenerationVerifierMetadataModel: Metadata for generation tasks
        - MolPropVerifierMetadataModel: Metadata for property prediction tasks
        - ReactionVerifierMetadataModel: Metadata for reaction tasks
    """

    rewards: list[float] = Field(
        ...,
        description="List of computed rewards for the molecular verification.",
    )
    parsed_answers: list[str] = Field(
        ..., description="The parsed answer extracted from the model completion."
    )
    verifier_metadatas: list[MolecularVerifierOutputMetadataModel] = Field(
        ...,
        description="List of metadata from each verifier used in the molecular verification.",
    )
