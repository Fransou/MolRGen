# Get Available RDKit Properties Tool

## `MolRGen_get_available_rdkit_properties`

Get a list of all available RDKit properties for molecular generation. Returns the RDKit property function names that can be evaluated by the molecular verifier.

**Returns:**
A sorted list of available RDKit property names (e.g., "SA", "QED", "CalcExactMolWt", "logP").

**Example Usage:**
```python
properties = get_available_rdkit_properties()
```
