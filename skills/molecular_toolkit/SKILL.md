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

Compute specific molecular properties for one or more SMILES strings.

**Arguments:**
- `smiles`: A list of SMILES strings to evaluate.
- `properties`: A list of property names to compute (e.g., ["QED", "logP", "SA"]).

**Returns:**
A dictionary where keys are SMILES strings and values are dictionaries mapping property names to their computed values.

**Example Usage:**
```python
result = get_properties(
    smiles=["CCO", "c1ccccc1"],
    properties=["QED", "logP", "SA"]
)
```
