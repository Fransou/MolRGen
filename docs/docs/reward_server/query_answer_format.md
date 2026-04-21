# Format of the Queries and Answers for the Molecular Verifier Server

All queries to the Molecular Verifier server and its responses follow specific formats defined using Pydantic models. Below we outline the expected structure for both requests and responses.

::: molrgen.server_utils.utils
    handler: python
    options:
        show_root_heading: false
        show_root_toc_entry: false
        members:
        - MolecularVerifierServerQuery
        - MolecularVerifierServerMetadata
        - MolecularVerifierServerResponse
        - BatchMolecularVerifierServerResponse
---
