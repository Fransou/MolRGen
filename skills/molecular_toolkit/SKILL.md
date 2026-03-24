---
name: molecular-toolkit
description: Tools for molecular generation, RDKit property calculation, and docking target analysis.
license: Apache-2.0
compatibility: Python 3.12+
user-invocable: true
mcp_servers:
  - MolRGen
allowed-tools:
  - read_file
  - grep
  - ask_user_question
  - write_file
---

# Molecular Toolkit

## When to use this skill
Use the `molecular_toolkit` skill when you need to perform tasks related to molecular generation, property calculation, or docking target identification. This skill provides a set of tools that can assist in various stages of molecular design and analysis.
With this skill you have enough chemical knowledge to understand the basics of chemistry and molecular modeling, and you can hence generate molecules, compute their properties, and perform multiple DMTA cycles (Design-Make-Test-Analyze) to optimize molecules for specific tasks.

## Available Tools through the MolRGen MCP Server

### `MolRGen_validate_query`

Validate a `MolecularVerifierServerQuery` for a specific task type. This tool ensures that a query is properly formatted and contains the necessary information for the specified task type.

**Arguments:**
- `query`: A `MolecularVerifierServerQuery` object or dictionary representation of the query to validate.
- `task`: The type of task to validate for. Options are:
  - "molecular_generation": Validates for molecule generation tasks
  - "property_prediction": Validates for property prediction tasks
  - "molecular_reaction": Validates for reaction planning tasks

**Returns:**
A validation result dictionary containing:
- "is_valid": bool indicating whether the query is valid
- "task": The task type that was validated
- "errors": List of error messages if validation failed
- "validated_query": The validated query object if valid

**Example Usage:**
```python
query = {
    "query": "Molecular SMILES",
    "metadata": [
        {
            "properties": ["logP"],
            "objectives": ["minimize"],
            "target": [0.0]
        }
    ]
}
result = validate_query(query=query, task="molecular_generation")
```

### `MolRGen_get_available_rdkit_properties`

Get a list of all available RDKit properties for molecular generation. returns the RDKit property function names that can be evaluated by the molecular verifier.

**Returns:**
A sorted list of available RDKit property names (e.g., "SA", "QED", "CalcExactMolWt", "logP").

**Example Usage:**
```python
properties = get_available_rdkit_properties()
```

### `MolRGen_get_available_docking_targets`

Get a list of available docking targets (receptors) for molecular generation. These targets can be used for docking simulations and property calculations.

**Returns:**
A dictionary mapping textual descriptions of the targets to their corresponding IDs.

**Example Usage:**
```python
targets = get_available_docking_targets()
```


### `MolRGen_get_properties`

Compute properties for one or more SMILES strings.

**Arguments:**
- `smiles`: List of SMILES strings to evaluate.
- `properties`: List of property names to compute (for example
    ``["QED", "logP", "SA"]``).

**Returns:**
Dict[str, Dict[str, float]]: A dictionary where keys are SMILES strings and values are dictionaries mapping property names to their computed values.

**Example Usage:**
```python
result = get_properties(
    smiles=["CCO", "c1ccccc1"],
    properties=["QED", "logP"]
)
```

### `MolRGen_train_reinvent_generator`

Train a REINVENT model with custom metadata for reward definition.

This endpoint runs the training synchronously and returns only after completion.

**Arguments:**
- `output_dir`: Output directory for model checkpoints (default: "./results")
- `num_train_epochs`: Number of training epochs (default: 10)
- `eval_batch_size`: Batch size for evaluation (default: 64)
- `batch_size`: Batch size for training (default: 64)
- `sigma`: Sigma parameter for REINVENT (default: 0.1)
- `learning_rate`: Learning rate for REINVENT (default: 1e-5)
- `smiles_start`: Beginning of sequence to start with (default: ["<s>"])
- `metadata`: Metadata defining the reward function and objectives (GenerationVerifierInputMetadataModel with properties, objectives, and target values)

**Returns:**
Dict[str, Any]: Response including:
    - `status`: 'completed' or 'failed'
    - `job_id`: Unique identifier for the training job
    - `message`: Status message
    - `output`: Training output
    - `timestamp`: Completion timestamp

**Example Usage:**
```python
from mol_gen_docking.reward.verifiers.generation_reward.input_metadata import GenerationVerifierInputMetadataModel

metadata = GenerationVerifierInputMetadataModel(
    properties=["QED", "SA"],
    objectives=["maximize", "minimize"],
    target=[0.8, 0.5]
)

params = ReinventTrainingParams(
    metadata=metadata,
    num_train_epochs=50,
    batch_size=64
)
result = train_reinvent_generator(params)
```

## Typical Molecular Generation Workflow

Typically, if tasked to generate molecules with specific properties, you would follow these steps:
1. **Find the corresponding objectives** Use `get_available_rdkit_properties` to find the RDKit properties, and `get_available_docking_targets` to find the docking targets that correspond to the desired objectives. Then formulate the query for the `MolecularVerifierServerQuery` accordingly, and validate it with `MolRGen_validate_query`.
2. **Train a REINVENT model** Use `MolRGen_train_reinvent_generator` to train a model with the validated query metadata. The training runs synchronously and waits for completion.
3. **Analyze results** Once the training is complete, analyze the generated molecules and their properties to ensure they meet the desired criteria.
4. **Iterate** If necessary, refine the query and repeat the process to further optimize the generated molecules for the specific task. You can notably run a second REINVENT training job while enforcing the beginning of the SMILES string to start with the best molecule from the previous training job, to further optimize it.
