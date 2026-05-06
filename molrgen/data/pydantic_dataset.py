import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Represents a single message in a conversation.

    A message is a single turn in a multi-turn conversation. It contains
    the speaker's role, the message content, and optional metadata.

    Attributes:
        role: The role of the message sender (system, user, or assistant).
        content: The text content of the message.
        meta: Optional metadata dictionary (e.g., timestamps, token counts).
        identifier: Optional unique identifier for this message.
        multimodal_document: Optional dictionary containing multimodal content
            like images, audio, or file attachments.

    Example:
        ```python
        message = Message(
            role="user",
            content="What is the capital of France?",
            meta={"tokens": 10}
        )
        ```
    """

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="The role of the sender: 'system', 'user', or 'assistant'"
    )
    content: str = Field(..., description="The text content of the message")
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata (timestamps, token counts, flags, etc.)",
    )
    identifier: Optional[str] = Field(
        None, description="Optional unique identifier for the message"
    )
    multimodal_document: Optional[Dict[str, Any]] = Field(
        None, description="Optional multimodal content (images, audio, files, etc.)"
    )


class Conversation(BaseModel):
    """Complete conversation structure with messages and metadata.

    This model represents a full conversation between a user and assistant,
    including all messages, system configuration, and evaluation metrics.
    The meta field should contain all information necessary to compute
    rewards, ground truth values, and other evaluation metrics.

    Attributes:
        meta: Metadata dictionary containing evaluation and reward computation data.
        messages: List of Message objects representing conversation turns.
        system_prompt: Optional system prompt that defines assistant behavior.
        available_tools: Optional list of tool names available to the assistant
            (e.g., 'calculator', 'web_search', 'code_interpreter').
        truncate_at_max_tokens: Optional maximum token limit for conversation.
        truncate_at_max_image_tokens: Optional maximum image token limit.
        output_modalities: Optional list of output types the model can produce
            (e.g., 'text', 'image', 'audio').
        identifier: Unique identifier for this conversation.
        references: Optional list of reference materials, ground truth data,
            or expected outputs for evaluation.
        rating: Optional quality rating or score (typically 0.0 to 1.0).
        source: Optional source or origin (e.g., 'human-generated', 'synthetic').
        training_masks_strategy: Strategy for applying attention masks during
            training (e.g., 'full', 'partial', 'causal').
        custom_training_masks: Optional custom mask configuration for
            advanced masking strategies.

    Example:
        ```python
        conversation = Conversation(
            meta={"task": "qa", "difficulty": "hard"},
            messages=[message1, message2],
            identifier="conv_001",
            training_masks_strategy="causal",
            source="human-generated"
        )
        ```
    """

    meta: Dict[str, Any] = Field(
        ..., description="Metadata for reward computation and evaluation (required)"
    )
    messages: List[Message] = Field(
        ..., description="List of messages in conversation order"
    )
    system_prompt: Optional[str] = Field(
        None, description="System prompt defining assistant behavior and constraints"
    )
    available_tools: Optional[List[str]] = Field(
        None, description="List of tools available for the assistant to use"
    )
    truncate_at_max_tokens: Optional[int] = Field(
        None, description="Maximum token count before truncation"
    )
    truncate_at_max_image_tokens: Optional[int] = Field(
        None, description="Maximum image token count before truncation"
    )
    output_modalities: Optional[List[str]] = Field(
        None, description="Supported output modalities (text, image, audio, etc.)"
    )
    identifier: str = Field(..., description="Unique conversation identifier")
    references: List[Any] = Field(
        default_factory=list,
        description="Reference data for evaluation and ground truth (LEGACY, contained in the metadata)",
    )
    rating: Optional[float] = Field(
        None, description="Quality rating (typically 0.0-1.0 scale)"
    )
    source: Optional[str] = Field(
        None, description="Source or origin of the conversation"
    )
    training_masks_strategy: str = Field(
        ..., description="Attention mask strategy during training"
    )
    custom_training_masks: Optional[Dict[str, Any]] = Field(
        None, description="Custom mask configuration"
    )


class Sample(BaseModel):
    """Root model containing all conversations for a single sample.

    A sample is the top-level container that groups one or more related
    conversations together with shared metadata and optional trajectories.
    This is typically the unit used for dataset serialization.

    Attributes:
        identifier: Unique identifier for this sample.
        conversations: List of Conversation objects in this sample.
        trajectories: Optional list of trajectories or execution paths
            for this sample (e.g., for reasoning or planning tasks).
        meta: Optional metadata about the sample (difficulty, domain, etc.).
        source: Optional source or origin (e.g., 'benchmark', 'user-submitted').

    Example:
        ```python
        sample = Sample(
            identifier="sample_001",
            conversations=[conversation1, conversation2],
            meta={"domain": "chemistry", "difficulty": "expert"},
            source="benchmark"
        )
        ```
    """

    identifier: str = Field(..., description="Unique sample identifier")
    conversations: List[Conversation] = Field(
        ..., description="List of conversations in this sample"
    )
    trajectories: List[Any] = Field(
        default_factory=list, description="Optional trajectories or execution paths"
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sample metadata (difficulty, domain, annotations, etc.)",
    )
    source: Optional[str] = Field(None, description="Source or origin of the sample")


def write_jsonl(output_file: Path, samples: list[Sample]) -> None:
    """Write Sample objects to a JSONL file.

    Each sample is serialized to JSON and written on a separate line.
    Parent directories are created automatically if they don't exist.

    Args:
        output_file: Path to the output JSONL file.
        samples: List of Sample objects to write.

    Example:
        ```python
        from pathlib import Path
        samples = [sample1, sample2, sample3]
        write_jsonl(Path("data/output.jsonl"), samples)
        ```
    """
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, "w") as fout:
        for sample in samples:
            fout.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")


def read_jsonl(input_file: Path) -> list[Sample]:
    """Read Sample objects from a JSONL file.

    Each line in the file should be a valid JSON object representing
    a Sample. Lines are parsed sequentially into Sample instances.

    Args:
        input_file: Path to the input JSONL file.

    Returns:
        List of Sample objects parsed from the file.

    Raises:
        AssertionError: If the input file does not exist.

    Example:
        ```python
        from pathlib import Path
        samples = read_jsonl(Path("data/input.jsonl"))
        for sample in samples:
            print(sample.identifier)
        ```
    """
    assert input_file.exists(), input_file

    samples: list[Sample] = []
    with open(input_file) as fin:
        for line in fin:
            samples.append(Sample(**json.loads(line)))

    return samples
