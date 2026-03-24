# Display Molecule Tool

## `MolRGen_display_molecule`

Generate a 2D ASCII art depiction of a molecule from a SMILES string using Open Babel.

**Arguments:**
- `smiles`: SMILES string to generate a 2D ASCII art representation

**Returns:**
A string containing the ASCII art representation of the molecule.

**Example Usage:**
```python
ascii_art = display_molecule("CCO")
print(ascii_art)
```
