# Validate Query Tool

## `MolRGen_validate_query`

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
