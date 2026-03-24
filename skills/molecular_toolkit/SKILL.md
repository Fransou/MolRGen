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

## Available Tools through the MolRGen MCP Server

For detailed documentation on each tool, see the followiing reference files if necessary:

- **[Validate Query Tool](references/validate_query.md)** - Query validation for molecular tasks
- **[Get Properties](references/get_properties.md)** - Compute properties for SMILES strings
- **[Training Tools](references/training.md)** - REINVENT model training and status monitoring
- **[Display Molecule Tool](references/display_molecule.md)** - 2D ASCII art visualization of molecules

The [properties](assets/properties.json) file contains a comprehensive list of molecular properties that can be computed using the `get_properties` tool, along with their descriptions and units.
Do not directly open it as it is a large file, but you can search for specific properties within it.

## Typical Molecular Generation Workflow

Typically, if tasked to generate molecules with specific properties, you would follow these steps:
1. **Find the corresponding objectives** for your task in the [properties](assets/properties.json) file.
2. **Start REINVENT training** Use `MolRGen_train_reinvent_generator` to start training a model with the validated query metadata. Note that reinvent training can take several minutes to run, depending on the parameters, especially when paired with docking-based objectives.
3. **Monitor training status** Use `MolRGen_get_training_status` to check the progress of your training job periodically (eg,every 30s).
4. **Analyze results** Once the training status shows 'completed', analyze the training's output files (generated molecules and training stats).
5. **Iterate** If necessary, refine the query and repeat the process to further optimize the generated molecules for the specific task. You can notably run a second REINVENT training job while enforcing the beginning of the SMILES string to start with.
6. **Visualize top molecules** Use `MolRGen_display_molecule` to display the 8 molecules with the highest score to the user along with their property values.
