"""Molecular verifier module for computing rewards across different task types.

This module provides the MolecularVerifier class which orchestrates reward computation
for molecular generation, property prediction, and chemical reaction tasks. It uses
Ray for parallel processing and supports GPU-accelerated docking calculations.
"""

import logging
from typing import Any, Callable, Dict, List

import ray
from rdkit import RDLogger

from molrgen.reward.molecular_verifier_pydantic_model import (
    BatchMolecularVerifierOutputModel,
    MolecularVerifierConfigModel,
    MolecularVerifierOutputMetadataModel,
)
from molrgen.reward.verifiers import (
    GenerationVerifier,
    GenerationVerifierOutputModel,
    MolPropVerifier,
    MolPropVerifierOutputModel,
    ReactionVerifier,
    ReactionVerifierOutputModel,
    assign_to_inputs,
)
from molrgen.reward.verifiers.abstract_verifier_pydantic_model import (
    BatchVerifiersInputModel,
)

RDLogger.DisableLog("rdApp.*")


class MolecularVerifier:
    """Main orchestrator for molecular verification and reward computation.

    This class provides a unified interface for computing rewards across different
    molecular tasks: de novo generation, property prediction, and chemical reactions.
    It automatically routes inputs to the appropriate verifier based on metadata.

    The verifier uses Ray for parallel processing and supports GPU-accelerated
    molecular docking calculations when configured.

    Attributes:
        verifier_config: Configuration model containing settings for all verifiers.
        logger: Logger instance for the verifier.

    Example:
        ```python
        from molrgen.reward import MolecularVerifier, MolecularVerifierConfigModel
        from molrgen.reward.verifiers import GenerationVerifierConfigModel

        config = MolecularVerifierConfigModel(
            generation_verifier_config=GenerationVerifierConfigModel(
                path_to_mappings="data/molgendata",
                reward="property"
            )
        )
        verifier = MolecularVerifier(config)

        # Compute rewards
        completions = ["<answer>CCO</answer>", "<answer>c1ccccc1</answer>"]
        metadata = [
            {"properties": ["QED"], "objectives": ["maximize"], "target": [0.0]},
            {"properties": ["SA"], "objectives": ["minimize"], "target": [0.0]}
        ]
        results = verifier(completions, metadata)
        print(f"Rewards: {results.rewards}")
        ```
    """

    def __init__(
        self,
        verifier_config: MolecularVerifierConfigModel,
    ):
        """Initialize the MolecularVerifier.

        Args:
            verifier_config: Configuration model containing settings for all
                sub-verifiers (generation, property prediction, reaction).
        """
        self.verifier_config = verifier_config
        self.__name__ = "RewardScorer/MolecularVerifier"

        self._generation_verifier: None | GenerationVerifier = None
        self._mol_prop_verifier: None | MolPropVerifier = None
        self._reaction_verifier: None | ReactionVerifier = None
        self.logger = logging.getLogger("RewardScorer")

        if not ray.is_initialized():
            ray.init()

    @property
    def generation_verifier(self) -> GenerationVerifier:
        """Lazy-loaded generation verifier instance.

        Returns:
            GenerationVerifier: The generation verifier for de novo molecular generation tasks.

        Raises:
            AssertionError: If generation_verifier_config is not set in the config.
        """
        if self._generation_verifier is not None:
            return self._generation_verifier
        assert self.verifier_config.generation_verifier_config is not None
        self._generation_verifier = GenerationVerifier(
            verifier_config=self.verifier_config.generation_verifier_config
        )
        return self._generation_verifier

    @property
    def mol_prop_verifier(self) -> MolPropVerifier:
        """Lazy-loaded molecular property verifier instance.

        Returns:
            MolPropVerifier: The verifier for molecular property prediction tasks.

        Raises:
            AssertionError: If mol_prop_verifier_config is not set in the config.
        """
        if self._mol_prop_verifier is not None:
            return self._mol_prop_verifier
        assert self.verifier_config.mol_prop_verifier_config is not None
        self._mol_prop_verifier = MolPropVerifier(
            verifier_config=self.verifier_config.mol_prop_verifier_config
        )
        return self._mol_prop_verifier

    @property
    def reaction_verifier(self) -> ReactionVerifier:
        """Lazy-loaded reaction verifier instance.

        Returns:
            ReactionVerifier: The verifier for chemical reaction and retro-synthesis tasks.

        Raises:
            AssertionError: If reaction_verifier_config is not set in the config.
        """
        if self._reaction_verifier is not None:
            return self._reaction_verifier
        assert self.verifier_config.reaction_verifier_config is not None
        self._reaction_verifier = ReactionVerifier(
            verifier_config=self.verifier_config.reaction_verifier_config
        )

        return self._reaction_verifier

    def _get_generation_score(
        self,
        inputs: BatchVerifiersInputModel,
        debug: bool = False,
        use_pbar: bool = False,
    ) -> List[GenerationVerifierOutputModel]:
        """Compute rewards for molecular generation tasks.

        Args:
            inputs: Batch of completions and metadata for generation verification.
            debug: If True, enables debug mode with additional logging.
            use_pbar: If True, displays a progress bar during computation.

        Returns:
            List of GenerationVerifierOutputModel containing rewards and metadata.
        """
        if debug:  # Testing purposes
            self.generation_verifier.debug = True
        elif self._generation_verifier is not None:
            self.generation_verifier.debug = False
        return self.generation_verifier.get_score(inputs)

    def _get_prop_pred_score(
        self,
        inputs: BatchVerifiersInputModel,
        debug: bool = False,
        use_pbar: bool = False,
    ) -> List[MolPropVerifierOutputModel]:
        """Compute rewards for molecular property prediction tasks.

        Args:
            inputs: Batch of completions and metadata for property verification.
            debug: If True, enables debug mode with additional logging.
            use_pbar: If True, displays a progress bar during computation.

        Returns:
            List of MolPropVerifierOutputModel containing rewards and metadata.
        """
        return self.mol_prop_verifier.get_score(inputs)

    def _get_reaction_score(
        self,
        inputs: BatchVerifiersInputModel,
        debug: bool = False,
        use_pbar: bool = False,
    ) -> List[ReactionVerifierOutputModel]:
        """Compute rewards for chemical reaction tasks.

        Args:
            inputs: Batch of completions and metadata for reaction verification.
            debug: If True, enables debug mode with additional logging.
            use_pbar: If True, displays a progress bar during computation.

        Returns:
            List of ReactionVerifierOutputModel containing rewards and metadata.
        """
        return self.reaction_verifier.get_score(inputs)

    def get_score(
        self,
        completions: List[Any],
        metadata: List[Dict[str, Any]],
        debug: bool = False,
        use_pbar: bool = False,
    ) -> BatchMolecularVerifierOutputModel:
        """Compute rewards for a batch of completions.

        This method automatically routes each completion to the appropriate verifier
        based on its metadata. It supports mixed batches containing generation,
        property prediction, and reaction tasks.

        Args:
            completions: List of model completions (strings or structured outputs).
            metadata: List of metadata dictionaries, one per completion. The metadata
                determines which verifier is used for each completion.
            debug: If True, enables debug mode with additional logging.
            use_pbar: If True, displays a progress bar during computation.

        Returns:
            BatchMolecularVerifierOutputModel containing:
                - rewards: List of float rewards for each completion
                - verifier_metadatas: List of metadata from each verification

        Raises:
            AssertionError: If completions and metadata have different lengths.

        Example:
            ```python
            completions = ["<answer>CCO</answer>"]
            metadata = [{"properties": ["QED"], "objectives": ["maximize"], "target": [0.0]}]
            results = verifier.get_score(completions, metadata)
            ```
        """
        assert len(completions) == len(metadata)
        obj_to_fn: Dict[
            str,
            Callable[
                [
                    BatchVerifiersInputModel,
                    bool,
                    bool,
                ],
                List[Any],
            ],
        ] = {
            "generation": self._get_generation_score,
            "mol_prop": self._get_prop_pred_score,
            "reaction": self._get_reaction_score,
        }
        idxs: Dict[str, List[int]] = {"generation": [], "mol_prop": [], "reaction": []}
        completions_per_obj: Dict[str, List[str]] = {
            "generation": [],
            "mol_prop": [],
            "reaction": [],
        }
        metadata_per_obj: Dict[str, List[Dict[str, Any]]] = {
            "generation": [],
            "mol_prop": [],
            "reaction": [],
        }
        for i, (completion, meta) in enumerate(zip(completions, metadata)):
            assigned = assign_to_inputs(completion, meta)
            idxs[assigned].append(i)
            completions_per_obj[assigned].append(completion)
            metadata_per_obj[assigned].append(meta)

        rewards = [0.0 for _ in range(len(metadata))]
        metadata_output = [
            MolecularVerifierOutputMetadataModel() for _ in range(len(metadata))
        ]
        parsed_answers: List[str] = ["" for _ in range(len(metadata))]
        for key, fn in obj_to_fn.items():
            if len(completions_per_obj[key]) > 0:
                outputs_obj = fn(
                    BatchVerifiersInputModel(
                        completions=completions_per_obj[key],
                        metadatas=metadata_per_obj[key],
                    ),
                    debug,
                    use_pbar,
                )
                for i, output in zip(idxs[key], outputs_obj):
                    rewards[i] = output.reward
                    metadata_output[i] = (
                        MolecularVerifierOutputMetadataModel.model_validate(
                            {f"{key}_verifier_metadata": output.verifier_metadata}
                        )
                    )
                    parsed_answers[i] = output.parsed_answer
        return BatchMolecularVerifierOutputModel(
            rewards=rewards,
            parsed_answers=parsed_answers,
            verifier_metadatas=metadata_output,
        )

    def __call__(
        self,
        completions: List[Any],
        metadata: List[Dict[str, Any]],
        debug: bool = False,
        use_pbar: bool = False,
    ) -> BatchMolecularVerifierOutputModel:
        """Call the verifier to compute rewards.

        This is a convenience method that calls get_score().

        Args:
            completions: List of model completions.
            metadata: List of metadata dictionaries.
            debug: If True, enables debug mode.
            use_pbar: If True, displays a progress bar.

        Returns:
            BatchMolecularVerifierOutputModel with rewards and metadata.
        """
        return self.get_score(
            completions=completions, metadata=metadata, debug=debug, use_pbar=use_pbar
        )
