import json
from copy import deepcopy
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from mol_gen_docking.data.pydantic_dataset import (
    Conversation,
    Message,
    Sample,
)
from mol_gen_docking.evaluation.diversity_aware_top_k import diversity_aware_top_k


class SFTExtractionConfig(BaseModel):
    """Configuration for SFT dataset extraction.

    Controls how completions are filtered, selected, and formatted into
    training samples. Supports reward thresholding, diversity-aware deduplication,
    and prompt enrichment via reward/source templates.

    Attributes:
        system_prompt_path: Path to a JSON file containing a system prompt template.
            If specified, the content replaces the system message in all conversations.
        min_reward_threshold: Minimum reward score a completion must achieve to be
            retained. Set to ``None`` to disable reward filtering.
        div_threshold: Tanimoto similarity threshold for diversity-aware filtering.
            Completions more similar than this threshold to an already-selected
            completion are discarded. Set to ``None`` to disable diversity filtering.
        fingerprint_name: Molecular fingerprint type used for diversity computation
            (e.g., ``"ecfp6-2048"``). Only relevant when ``div_threshold`` is set.
        reward_info_template: Per-role templates for injecting reward information into
            prompt messages. Keys are message roles (e.g., ``"system"``), values are
            format strings with ``{content}`` and ``{reward}`` placeholders.
        source_info_template: Per-role templates for injecting source information into
            prompt messages. Keys are message roles, values are format strings with
            ``{content}`` and ``{source}`` placeholders. Applied after
            ``reward_info_template``.

    Example:
        ```python
        config = SFTExtractionConfig(
            min_reward_threshold=0.5,
            div_threshold=0.7,
            fingerprint_name="ecfp4-1024",
            reward_info_template={
                "system": "{content}\\nPropose an answer whose reward is: {reward:.2f}"
            },
            source_info_template={},
        )
        ```
    """

    system_prompt_path: Optional[str] = Field(
        None,
        description="Path to a text file containing the system prompt template. \
            If specified, the content of this file will be used as the system prompt template for all \
            conversations during SFT extraction. The template should include a placeholder for the content (e.g., {content}).",
    )
    min_reward_threshold: Optional[float] = Field(
        ...,
        description="Minimum reward threshold for filtering completions during SFT extraction.",
    )
    div_threshold: Optional[float] = Field(
        ...,
        description="Diversity threshold for diversity-aware top-k selection during SFT extraction.",
    )
    fingerprint_name: Optional[str] = Field(
        "ecfp6-2048",
        description="Fingerprint type to use for diversity-aware top-k selection. \
            This should be a valid fingerprint type supported by the underlying implementation.",
    )
    reward_info_template: Dict[str, str] = Field(
        default_factory=lambda x: dict(
            user="{content}\n(Propose an answer whose reward is: {reward:.2f})"
        ),
        description="Template used to add the reward information to the prompt (specified by the key). \
            The template should include placeholders for the content and reward values.",
    )
    source_info_template: Dict[str, str] = Field(
        default_factory=lambda x: dict(user="{content}\n(Source model: {source})"),
        description="Template used to add the source information to the prompt (specified by the key). \
            The template should include placeholders for the content and source values. \
            This pattern is always applied after the reward_info_template if both are specified.",
    )


class SFTExtractionMetadata(BaseModel):
    """Metadata accumulated during SFT extraction.

    Tracks per-completion statistics (prompt IDs, rewards, estimated token counts)
    for all completions that passed filtering and were included in the final dataset.

    Attributes:
        prompt_ids: Prompt identifiers for each retained completion, preserving
            the order in which completions were processed.
        rewards: Reward scores for each retained completion. Defaults to ``0.0``
            when the original reward is ``None``.
        n_tokens: Rough token-count estimates for each retained completion,
            computed as ``len(output) // 4``.
    """

    prompt_ids: List[str] = Field(
        default_factory=list,
        description="List of prompt IDs associated with the completions. \
            Used to group completions that were generated from the same prompt for filtering during SFT extraction.",
    )
    rewards: List[float] = Field(
        default_factory=list, description="List of rewards extracted for each prompt."
    )
    n_tokens: List[int] = Field(
        default_factory=list,
        description="List of token counts for each completion (number of characters divided by 4 as a rough estimate).",
    )


class Completion(BaseModel):
    """A single model completion with associated reward and metadata.

    Represents one generated output for a given prompt, together with the
    reward score assigned by the reward model and any verifier metadata
    needed for correctness filtering and diversity-aware selection.

    Attributes:
        output: The raw generated text (e.g., containing ``<answer>...</answer>`` tags).
        reward: Reward score assigned to this completion, or ``None`` if not yet scored.
        metadata: Arbitrary metadata dict; must contain a ``"prompt_id"`` key that
            links the completion back to its originating prompt.
        reward_meta: Verifier metadata produced by the reward pipeline. Expected to
            contain one of ``generation_verifier_metadata``,
            ``mol_prop_verifier_metadata``, or ``reaction_verifier_metadata``.
        source: Optional label identifying the model or method that produced the
            completion (e.g., ``"Qwen3-0.6B-sft-v2"``).

    Example:
        ```python
        comp = Completion(
            output="<answer>CCO</answer>",
            reward=0.85,
            metadata={"prompt_id": "prompt_42"},
            reward_meta={
                "generation_verifier_metadata": {"all_smi": ["CCO"]}
            },
            source="my_model_v1",
        )
        ```
    """

    output: str = Field(
        ...,
        description="The generated completion text.",
    )
    reward: Optional[float] = Field(
        None,
        description="The reward score associated with the completion.",
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Metadata associated with the completion, including generation verifier metadata and other relevant information.",
    )
    reward_meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata related to the reward computation.",
    )
    source: Optional[str] = Field(
        None,
        description="The source of the completion (e.g., model name, generation method).",
    )


class SFTExtractor:
    """Extracts SFT training samples from a collection of completions and prompts.

    Orchestrates the full extraction pipeline: correctness filtering, per-prompt
    grouping, diversity-aware selection, reward thresholding, conversation
    formatting, and post-processing (reward/source injection).

    Attributes:
        config: The `SFTExtractionConfig` controlling extraction behaviour.
        metadata: An `SFTExtractionMetadata` instance accumulating statistics
            about retained completions across calls to :meth:`extract`.

    Example:
        ```python
        from mol_gen_docking.evaluation.sft_extraction import (
            SFTExtractionConfig, SFTExtractor, Completion,
        )

        config = SFTExtractionConfig(
            min_reward_threshold=0.5,
            div_threshold=None,
            reward_info_template={},
            source_info_template={},
        )
        extractor = SFTExtractor(config)
        samples = extractor.extract(completions, prompts)
        ```
    """

    def __init__(self, config: SFTExtractionConfig) -> None:
        """Initialise the SFT extractor.

        Args:
            config: Extraction configuration controlling filtering thresholds,
                diversity parameters, and prompt templates.
        """
        self.config = config
        self.metadata = SFTExtractionMetadata()

    def filter_is_correct(self, completions: List[Completion]) -> List[Completion]:
        """Filter completions to keep only those that are semantically correct.

        Automatically detects the task type from the verifier metadata and applies
        the appropriate correctness check:

        - **Molecular generation** (``generation_verifier_metadata``): keeps
          completions with exactly one extracted SMILES.
        - **Property prediction** (``mol_prop_verifier_metadata``): keeps
          completions where value extraction succeeded.
        - **Retro-synthesis** (``reaction_verifier_metadata``): keeps completions
          whose validity score is strictly positive.

        If any completion has empty ``reward_meta``, all completions are returned
        unfiltered (graceful fallback).

        Args:
            completions: List of completions to filter.

        Returns:
            A new list containing only the completions that passed the
            task-specific correctness check.

        Raises:
            ValueError: If none of the known verifier metadata keys are found
                in the completions.
        """
        if any(len(c.reward_meta) == 0 for c in completions):
            return completions
        # First if molecular generation:
        if all(
            c.reward_meta.get("generation_verifier_metadata", None) is not None
            for c in completions
        ):
            return [
                c
                for c in completions
                if len(c.reward_meta["generation_verifier_metadata"]["all_smi"]) == 1
            ]
        # Molecular property prediction
        if all(
            c.reward_meta.get("mol_prop_verifier_metadata", None) is not None
            for c in completions
        ):
            return [
                c
                for c in completions
                if c.reward_meta["mol_prop_verifier_metadata"]["extraction_success"]
            ]
        if all(
            c.reward_meta.get("reaction_verifier_metadata", None) is not None
            for c in completions
        ):
            return [
                c
                for c in completions
                if c.reward_meta["reaction_verifier_metadata"]["valid"] > 0.0
            ]
        raise ValueError("Unknown verifier metadata format for filtering completions.")

    def filter_completions_single_id(
        self, completions: List[Completion]
    ) -> List[Completion]:
        """Filter and rank completions that share a single prompt ID.

        Applies the following filters in order:

        1. **Diversity-aware selection** — if ``config.div_threshold`` is set, uses
           :func:`~mol_gen_docking.evaluation.diversity_aware_top_k.diversity_aware_top_k`
           to remove chemically redundant completions.
        2. **Reward threshold** — if ``config.min_reward_threshold`` is set, discards
           completions whose reward is below the threshold.

        Side-effects: appends per-completion statistics (reward, token count,
        prompt ID) to ``self.metadata``.

        Args:
            completions: List of completions that must all share the same
                ``metadata["prompt_id"]``.

        Returns:
            The filtered list of completions.

        Raises:
            AssertionError: If completions do not share the same prompt ID, or if
                diversity filtering is requested but reward metadata is missing.
        """
        prompt_id = completions[0].metadata["prompt_id"]
        assert all(c.metadata["prompt_id"] == prompt_id for c in completions), (
            "All completions must have the same prompt_id for filtering."
        )

        # Filter completions based on diversity-aware top-k selection
        if self.config.div_threshold is not None:
            assert all(
                len(c.reward_meta) > 0 and c.reward is not None for c in completions
            ), (
                "Reward metadata must be available for all completions when min_reward_threshold is set."
            )
            assert all(
                c.reward_meta["generation_verifier_metadata"] is not None
                for c in completions
            ), (
                "Diversity aware sampling can only be performed on molecular generation tasks"
            )

            if len(completions) > 1:
                _, keep_indices = diversity_aware_top_k(
                    mols=[
                        comp.reward_meta["generation_verifier_metadata"]["all_smi"][0]
                        for comp in completions
                    ],
                    scores=[c.reward for c in completions],  # type: ignore
                    k=len(completions),
                    t=self.config.div_threshold,
                    fingerprint_name=self.config.fingerprint_name,
                    return_idxs=True,
                )
                completions = [completions[i] for i in keep_indices]

        # Filter completions based on minimum reward threshold
        if self.config.min_reward_threshold is not None:
            assert all(c.reward is not None for c in completions)
            completions = [
                c
                for c in completions
                if c.reward is not None and c.reward >= self.config.min_reward_threshold
            ]

        # Update metadata
        for c in completions:
            if c.reward is not None:
                self.metadata.rewards.append(c.reward)
            else:
                self.metadata.rewards.append(0.0)
            self.metadata.n_tokens.append(
                len(c.output) // 4
            )  # Rough estimate of token count
            self.metadata.prompt_ids.append(prompt_id)

        return completions

    def completion_to_conv(
        self,
        completions: List[Completion],
        prompt: Sample,
    ) -> Sample | None:
        """Convert filtered completions into a multi-conversation Sample.

        For each completion, creates a new `Conversation` by copying the
        prompt messages and appending the completion text as an assistant message.
        Metadata such as ``source``, ``rating``, and ``training_masks_strategy``
        are carried over from the original conversation / completion.

        Args:
            completions: Filtered completions to convert. All must have
                ``metadata["prompt_id"]`` equal to ``prompt.identifier``.
            prompt: The original prompt `Sample` whose first conversation
                provides the base messages.

        Returns:
            A new `Sample` containing one `Conversation` per
            completion, or ``None`` if an error occurs.

        Raises:
            AssertionError: If any completion's prompt ID does not match
                ``prompt.identifier``.
        """
        assert all(c.metadata["prompt_id"] == prompt.identifier for c in completions), (
            "All completions must have the same prompt_id as the prompt sample for conversion."
        )

        conversation = prompt.conversations[0]
        new_conversation_messages = []
        for completion in completions:
            # stop the completion after the last </answer> tag
            assistant_content = completion.output
            assistant_content = assistant_content.split("</answer>")[0] + "</answer>"
            assistant_message = Message(role="assistant", content=assistant_content)
            prompt_message = deepcopy(conversation.messages)
            prompt_message.append(assistant_message)
            new_conversation = Conversation(
                messages=prompt_message,
                source=completion.source,
                rating=completion.reward,
                meta=conversation.meta,
                identifier=conversation.identifier,
                training_masks_strategy=conversation.training_masks_strategy,
            )
            new_conversation_messages.append(new_conversation)
        new_sample = Sample(
            conversations=new_conversation_messages,
            identifier=conversation.identifier,
        )
        return new_sample

    def post_process_sample(self, sample: Sample) -> Sample:
        """Enrich a sample's messages with reward and source information.

        Iterates over every conversation and message in *sample* and applies:

        1. ``config.reward_info_template`` — injects the conversation's reward
           into matching messages (keyed by role).
        2. ``config.source_info_template`` — injects the conversation's source
           into matching messages (keyed by role). Always applied **after** the
           reward template.

        Args:
            sample: The `Sample` to post-process. Modified **in-place**.

        Returns:
            The same `Sample` instance, with message contents updated.
        """
        # Add reward information to the system prompt if a template is provided
        if self.config.reward_info_template:
            for conv in sample.conversations:
                for msg in conv.messages:
                    if msg.role in self.config.reward_info_template:
                        reward = conv.rating if conv.rating is not None else 0.0
                        msg.content = self.config.reward_info_template[msg.role].format(
                            content=msg.content,
                            reward=reward,
                        )
        # Add source information to the system prompt if a template is provided
        if self.config.source_info_template:
            for conv in sample.conversations:
                for msg in conv.messages:
                    if msg.role in self.config.source_info_template:
                        source = conv.source if conv.source is not None else "unknown"
                        msg.content = self.config.source_info_template[msg.role].format(
                            content=msg.content,
                            source=source,
                        )
        return sample

    def get_sample(
        self, completions: List[Completion], prompt: Sample
    ) -> Sample | None:
        """Build a single SFT sample from completions and their originating prompt.

        Convenience method that chains :meth:`filter_completions_single_id`,
        :meth:`completion_to_conv`, and :meth:`post_process_sample`.

        Args:
            completions: Completions for one prompt ID.
            prompt: The corresponding prompt `Sample`.

        Returns:
            A fully processed `Sample` ready for SFT training, or ``None``
            if no completions survive filtering.
        """
        filtered_completions = self.filter_completions_single_id(completions)
        if len(filtered_completions) == 0:
            return None
        sample = self.completion_to_conv(filtered_completions, prompt)
        if sample is not None:
            sample = self.post_process_sample(sample)
        return sample

    def extract(
        self, completions: List[Completion], prompts: List[Sample]
    ) -> List[Sample]:
        """Run the full SFT extraction pipeline.

        Executes the following steps:

        1. **Correctness filtering** — removes invalid completions via
           :meth:`filter_is_correct`.
        2. **Grouping** — groups surviving completions by ``prompt_id``.
        3. **System prompt override** — if ``config.system_prompt_path`` is set,
           replaces or inserts the system message in every prompt conversation.
        4. **Per-prompt extraction** — calls :meth:`get_sample` for each prompt
           that has at least one completion.

        Args:
            completions: All completions across all prompts.
            prompts: The original prompt `Sample` objects. Only prompts
                whose ``identifier`` appears in the completions will produce
                output samples.

        Returns:
            List of `Sample` objects suitable for SFT training. Each
            sample may contain multiple conversations (one per retained
            completion).
        """
        samples = []
        completions = self.filter_is_correct(completions)

        id_to_completions: Dict[str, List[Completion]] = {}
        for c in tqdm(completions, desc="Grouping completions by prompt_id"):
            prompt_id = c.metadata["prompt_id"]
            if prompt_id not in id_to_completions:
                id_to_completions[prompt_id] = []
            id_to_completions[prompt_id].append(c)
        for prompt in tqdm(prompts, desc="Processing prompts and extracting samples"):
            # Add the system prompt template to the prompt conversations if specified in the config
            if self.config.system_prompt_path is not None:
                with open(self.config.system_prompt_path) as f:
                    system_prompt = json.load(f).copy()
                for conv in prompt.conversations:
                    if conv.messages[0].role == "system":
                        conv.messages[0].content = system_prompt["content"]
                    else:
                        conv.messages.insert(
                            0, Message(role="system", content=system_prompt["content"])
                        )

            prompt_id = prompt.identifier
            if prompt_id not in id_to_completions:
                continue
            sample = self.get_sample(id_to_completions[prompt_id], prompt)
            if sample is not None:
                samples.append(sample)

        return samples
