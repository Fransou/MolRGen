# Get Properties Tool

## `MolRGen_get_properties`

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
