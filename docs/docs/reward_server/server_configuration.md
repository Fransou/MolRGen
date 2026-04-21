# Configuration of the Molecular Verifier Server

## Using a Pydantic BaseSettings

We use Pydantic's `BaseSettings` to manage the configuration of our application through environment variables. It automatically reads and validates environment variables, converting them to Python types defined in your settings class.


**How it works:**
When you define a class that inherits from `BaseSettings`, Pydantic:

1. Looks for environment variables matching the field names (case-insensitive)
2. Converts them to the appropriate Python types
3. Validates them against the field constraints
4. Falls back to default values if environment variables are not provided

**Setting configuration:**
Simply export environment variables before starting the server:
```bash
export DOCKING_ORACLE="autodock_gpu"
export DATA_PATH="/path/to/data"
uvicorn ...
```

Or set them inline:
```bash
DOCKING_ORACLE=autodock_gpu DATA_PATH=./data uvicorn ...
```

## Server Configuration

::: molrgen.server_utils.server_setting
    handler: python
    options:
        show_root_heading: false
        show_root_toc_entry: false
        members:
        - MolecularVerifierServerSettings
---
