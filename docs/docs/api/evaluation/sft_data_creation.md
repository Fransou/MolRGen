# SFT Dataset Creation

## Overview

The **SFT extraction** module provides tools for creating **Supervised Fine-Tuning (SFT) datasets** from generated completions. It filters, ranks, and formats model completions into structured conversation samples suitable for training, with support for reward-based filtering, diversity-aware selection, and customizable prompt templates.

## Key Concepts

### Extraction Pipeline

The SFT extraction pipeline follows these steps:

1. **Correctness filtering** — Remove completions that are invalid (e.g., malformed SMILES, failed extractions)
2. **Grouping** — Group completions by their originating prompt ID
3. **Diversity-aware filtering** *(optional)* — Deduplicate chemically similar completions using molecular fingerprints (see [Diversity-Aware Top-k](diversity_aware_top_k.md))
4. **Reward threshold filtering** *(optional)* — Keep only completions whose reward exceeds a minimum threshold
5. **Post-processing** — Optionally inject reward and source information into the prompt messages

### Reward & Source Templating

After building conversation samples, the extractor can enrich prompts with **reward** and **source** information using configurable templates. For example, with the default `reward_info_template`:

```python
# Original system message:
"Generate a molecule with high binding affinity."

# After reward templating (reward=0.85):
"Generate a molecule with high binding affinity.\nPropose an answer whose reward is: 0.85"
```


## Usage Examples

### Basic SFT Extraction

```python
from mol_gen_docking.evaluation.sft_extraction import (
    SFTExtractionConfig,
    SFTExtractor,
    Completion,
)
from mol_gen_docking.data.pydantic_dataset import Sample, Conversation, Message

# Define the extraction configuration
config = SFTExtractionConfig(
    min_reward_threshold=0.5,
    div_threshold=None,  # No diversity filtering
    reward_info_template={},  # No reward injection
    source_info_template={},  # No source injection
)

extractor = SFTExtractor(config)

# Build a prompt sample
prompt = Sample(
    identifier="prompt_0",
    conversations=[
        Conversation(
            messages=[
                Message(role="system", content="You are a molecular generation assistant."),
                Message(role="user", content="Generate a molecule with high docking score."),
            ]
        )
    ],
)

# Build completions (e.g., from a generation run)
completions = [
    Completion(
        output="<answer>CCO</answer>",
        reward=0.8,
        metadata={"prompt_id": "prompt_0"},
        reward_meta={
            "generation_verifier_metadata": {"all_smi": ["CCO"]}
        },
        source="my_model_v1",
    ),
    Completion(
        output="<answer>invalid</answer>",
        reward=0.1,
        metadata={"prompt_id": "prompt_0"},
        reward_meta={
            "generation_verifier_metadata": {"all_smi": ["invalid"]}
        },
        source="my_model_v1",
    ),
]

# Extract SFT samples
samples = extractor.extract(completions, [prompt])
print(len(samples))
# Output:
# >>> 1
```

### With Diversity-Aware Filtering

```python
config = SFTExtractionConfig(
    min_reward_threshold=0.3,
    div_threshold=0.7,            # Tanimoto similarity threshold
    fingerprint_name="ecfp4-1024",
    reward_info_template={},
    source_info_template={},
)

extractor = SFTExtractor(config)
samples = extractor.extract(completions, [prompt])
```

### With Reward-Conditioned Prompts

```python
config = SFTExtractionConfig(
    min_reward_threshold=0.5,
    div_threshold=None,
    reward_info_template={
        "system": "{content}\nPropose an answer whose reward is: {reward:.2f}"
    },
    source_info_template={
        "system": "{content}\nThe source of this conversation is: {source}"
    },
)

extractor = SFTExtractor(config)
samples = extractor.extract(completions, [prompt])

# The system message of each conversation now includes reward and source info
print(samples[0].conversations[0].messages[0].content)
# Output:
# >>> "You are a molecular generation assistant.
# >>> Propose an answer whose reward is: 0.80
# >>> The source of this conversation is: my_model_v1"
```

### Using a Custom System Prompt File

```python
config = SFTExtractionConfig(
    system_prompt_path="system_prompts/vanilla.json",
    min_reward_threshold=0.5,
    div_threshold=None,
    reward_info_template={},
    source_info_template={},
)

extractor = SFTExtractor(config)
# The system prompt from the JSON file will replace the existing system message
samples = extractor.extract(completions, [prompt])
```

### Inspecting Extraction Metadata

After extraction, the `SFTExtractor` tracks metadata about the retained completions:

```python
extractor = SFTExtractor(config)
samples = extractor.extract(completions, [prompt])

# Prompt IDs of all retained completions
print(extractor.metadata.prompt_ids)

# Rewards of retained completions
print(extractor.metadata.rewards)

# Estimated token counts
print(extractor.metadata.n_tokens)
```

## Class & Function Reference

::: mol_gen_docking.evaluation.sft_extraction
    options:
        show_root_toc_entry: false
        heading_level: 3
