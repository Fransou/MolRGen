# Data Access

## Overview

MolRGen addresses the challenge of *de novo* molecular generation, with a benchmark designed for Large Language Models (LLMs) and other generative architectures.
Our dataset currently supports 3 downstream tasks:

| Dataset                            | Size           | Source                                                                                                       | Purpose                                                            |
|------------------------------------|----------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| ***De Novo* Molecular Generation** | ~50k prompts   | [SAIR](https://huggingface.co/datasets/SandboxAQ/SAIR) and [SIU](https://huggingface.co/datasets/bgao95/SIU) | Generate molecules optimizing a set of up to three properties      |
| **Molecular Property prediction**  | ~50k prompts   | [Polaris](https://polarishub.io/)                                                                            | Predict the properties of a molecule (regression + classification) |
| **RetroSynthesis Tasks**           | ~50k reactions | [Enamine](https://enamine.net/building-blocks/building-blocks-catalog)                                       | Retro-synthesis planning, reactants, products, SMARTS prediction   |


## Downloading Datasets
Our dataset is available on [huggingface.](https://huggingface.co/datasets/Franso/MolGenData)
The three tasks are separated into three compressed folders: `molgendata.tar.gz`, `property_predictions.tar.gz`, and `synthesis_tasks.tar.gz`.
HF datasets are also available, but do not include the metadatas necessary for the reward computation, and are thus only meant for inference.

```bash
# Extract tarball datasets
tar -xzf molgendata.tar.gz
tar -xzf property_prediction.tar.gz
tar -xzf synthesis_tasks.tar.gz
```
!!! warning
    To perform *de novo* molecular generation with docking constraints, the path to the `molgendata` folder should be provided to the reward server via the `DATA_PATH` environment variable. (See [Reward Server Configuration](../reward_server/getting_started.md) for more details.)

## Data Organization

=== "Molecular Generation Data"
    ```plaintext
    molgendata/              # Docking targets
    ├── docking_targets.json
    ├── names_mapping.json
    ├── pockets_info.json
    ├── pdb_files/
    ├── train_data/
    │   └── train_prompts.jsonl
    │   └── train_prompts_boxed.jsonl
    ├── eval_data/
    │   └── eval_prompts_ood_boxed.jsonl
    │   └── eval_prompts_ood.jsonl
    └── test_data/
        └── test_prompts_ood_boxed.jsonl
        └── test_prompts_ood.jsonl
    ```
=== "Property Prediction Data"
    ```plaintext
    property_prediction/               # Reaction data
    ├── train_prompts_boxed.jsonl
    └── eval_prompts_ood_boxed.jsonl
    ```
=== "Retro-Synthesis Data"
    ```plaintext
    synthesis_tasks/                 # Multi-task benchmarks
    ├── train_prompts.jsonl
    └── eval_prompts
        └── eval_prompts_[chembl/enamine]_[#building blocks].jsonl
    ```


### Data Format

Our data are stored in JSONL format, represented by a pydantic base model ([source](https://github.com/Fransou/MolRGen/blob/main/molrgen/data/pydantic_dataset.py)):

```python
class Message(BaseModel):
    """Represents a single message in a conversation"""

    role: Literal["system", "user", "assistant"]
    content: str
    meta: Dict[str, Any] = Field(default_factory=dict)
    identifier: Optional[str] = None
    multimodal_document: Optional[Dict[str, Any]] = None


class Conversation(BaseModel):
    """Complete conversation structure"""

    meta: Dict[
        str, Any
    ]
    messages: List[Message]
    system_prompt: Optional[str] = None
    available_tools: Optional[List[str]] = None
    truncate_at_max_tokens: Optional[int] = None
    truncate_at_max_image_tokens: Optional[int] = None
    output_modalities: Optional[List[str]] = None
    identifier: str
    references: List[Any] = Field(default_factory=list)
    rating: Optional[float] = None
    source: Optional[str] = None
    training_masks_strategy: str
    custom_training_masks: Optional[Dict[str, Any]] = None


class Sample(BaseModel):
    """Root model containing all conversations"""

    identifier: str
    conversations: List[Conversation]
    trajectories: List[Any] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None
```

**JSONL Format**

Each line in a JSONL file represents a `Sample` object with the following structure:

=== "Molecular Generation Sample Example"
    ```json
    {
      "identifier": "sample_001",
      "conversations": [
        {
          "meta": {
            "properties": ["QED", ...],
            "objectives": ["above", ...],
            "target": [0.5, ...],
            ... # Additional metadata such as pocket box, target information, etc.
          },
          "messages": [
            {
              "role": "system",
              "content": "You are a molecular generation assistant...",
              "meta": {}
            },
            {
              "role": "user",
              "content": "Generate a molecule that binds to GSK3B with high affinity",
              "meta": {},
              "identifier": "msg_001"
            }
          ]
        }
      ]
    }
    ```
=== "Property Prediction Sample Example"
    ```json
    {
      "identifier": "id0",
      "conversations": [
        {
          "meta": {
            "properties": ["biogen/adme-fang-hclint-reg-v1"],
            "objectives": ["regression"],
            "target": [0.67],
            "smiles": ["Brc1ccc(-c2nnc(Cn3cnc4ccccc43)o2)o1"],
            "norm_var": 0.62,
            ... # Additional metadata
          },
          "messages": [
            {
              "role": "system",
              "content": "You are a property prediction assistant...",
            },
            {
              "role": "user",
              "content": "Predict the clearance value for the molecule with SMILES: Brc1ccc(-c2nnc(Cn3cnc4ccccc43)o2)o1",
            }
          ]
        }
      ]
    }
    ```
=== "Retro-Synthesis Sample Example"
    ```json
    {
      "identifier": "id0",
      "conversations": [
        {
          "meta": {
            "properties": ["full_path_bb_ref"],
            "objectives": ["full_path_bb_ref"],
            "target": ["CCCC[C@H](NC(=O)[C@H](CCCCN)NC(=O)[C@H](CCCNC(=N)N)NC(=O)c1ccc(/C=C2\\SC(=O)N(c3ccc(C)cc3)C2=O)cc1)C(N)=O"],
            ... # Additional metadata
          },
          "messages": [
            {
              "role": "system",
              "content": "You are a retrosynthesis planning assistant...",
            },
            {
              "role": "user",
              "content": "Given the product molecule with SMILES: CC(=O)OC1=CC=CC=C1C(=O)O, provide the reactants used in its synthesis.",
            }
          ]
        }
      ]
    }
    ```
