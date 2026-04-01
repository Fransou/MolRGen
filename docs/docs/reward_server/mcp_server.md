# MCP Server for Molecular Verification

The Molecular Verifier Server includes MCP (Model Context Protocol) support, enabling integration with language models and AI workflows that use the MCP standard.

## Overview

The MCP server provides the same molecular verification capabilities as the REST API, but through an MCP-compatible interface. This allows seamless integration with AI systems that use MCP for tool calling and function execution.

## Starting the MCP Server

To start the server with MCP support:

```bash
export DOCKING_ORACLE=autodock_gpu
export DATA_PATH=data
python mol_gen_docking/server_mcp.py
```

You should see output indicating both FastAPI and MCP server initialization:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

## MCP Tools

The server exposes several MCP tools for molecular verification:

<div class="grid cards" markdown>

-   :material-flask-outline:{ .lg .middle } __Query Validation__

    ---

    **Tool:** `validate_query`

    **Endpoint:** `POST /validate_query/{task}`

    **Description:** Validates a query for a specific task type (molecular generation, property prediction, or molecular reaction).

    **Parameters:**

    - `query`: MolecularVerifierServerQuery object or dictionary
    - `task`: Task type ("molecular_generation", "property_prediction", "molecular_reaction")

    **Returns:** Validation result with is_valid flag, errors, and validated query.

-   :material-code-braces:{ .lg .middle } __Reward Calculation__

    ---

    **Tool:** `compute_reward`

    **Endpoint:** `POST /get_reward_mcp`

    **Description:** Computes reward scores for molecular generations based on specified properties and objectives.

    **Parameters:**

    - `query`: MolecularVerifierServerQuery containing completions and metadata

    **Returns:** BatchMolecularVerifierServerResponse with rewards and metadata.

-   :material-sitemap:{ .lg .middle } __Property Calculation__

    ---

    **Tool:** `get_properties`

    **Endpoint:** `POST /get_properties`

    **Description:** Computes specific properties for given SMILES strings.

    **Parameters:**

    - `smiles`: List of SMILES strings
    - `properties`: List of property names to compute

    **Returns:** Dictionary mapping SMILES to property values.

-   :material-robot:{ .lg .middle } __REINVENT Training__

    ---

    **Tool:** `train_reinvent_generator`

    **Endpoint:** `POST /train_reinvent_generator`

    **Description:** Train a REINVENT model with custom metadata for reward definition.

    **Parameters:**

    - `params`: Training parameters including model configuration and metadata

    **Returns:** Response including status, job_id, message, and timestamp.

-   :material-progress-check:{ .lg .middle } __Training Status__

    ---

    **Tool:** `get_training_status`

    **Endpoint:** `POST /get_training_status/{job_id}`

    **Description:** Get the status of a REINVENT training job.

    **Parameters:**

    - `job_id`: The job identifier returned by train_reinvent_generator

    **Returns:** Job status information including status, timestamps, and output.

-   :material-eye:{ .lg .middle } __Molecule Visualization__

    ---

    **Tool:** `display_molecule`

    **Endpoint:** `POST /display_molecule/{smiles}`

    **Description:** Generate a 2D ASCII art depiction of a molecule from a SMILES string.

    **Parameters:**

    - `smiles`: SMILES string to visualize

    **Returns:** ASCII art representation of the molecule.

</div>



## Troubleshooting

For optimal performance with the MCP server, ensure the following configuration:

- **Batch Mode**: The server should be configured to use batch processing mode for efficient handling of multiple requests.
- **Low Buffer Time**: Set a low buffer time to minimize latency in processing requests.
- **Parsing Method**: The parsing method must be set to `none` to ensure compatibility with the MCP protocol.

See [Server Configuration](server_configuration.md) for performance tuning options.
