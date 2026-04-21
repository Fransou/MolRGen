from typing import Any, Dict, List, cast

import requests


def get_data_from_pdb_id(pdb_id: str) -> Dict[str, List[Dict[str, Any]]]:
    pdb_id = pdb_id.replace("_docking", "")
    url = f"https://rest.uniprot.org/uniprotkb/search?query=(xref:pdb-{pdb_id})"

    response = requests.get(url, timeout=20)
    if response.status_code != 200:
        raise ValueError(f"Failed to query UniProt for PDB ID {pdb_id}")

    data: Dict[str, List[Dict[str, Any]]] = response.json()
    return data


def fetch_uniprot_id_from_pdbid(pdb_id: str) -> str | None:
    """
    Fetches the UniProt accession ID corresponding to a PDB ID using the UniProt API.
    """
    data = get_data_from_pdb_id(pdb_id)
    if not data.get("results"):
        raise ValueError(f"No UniProt mapping found for PDB ID {pdb_id}")
    uniprot_id = cast(str, data["results"][0]["primaryAccession"])

    return uniprot_id


def fetch_uniprot_info(uniprot_id: str) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    for i in range(5):
        try:
            response = requests.get(url, timeout=180)
        except Exception as e:
            print("Error with uniprot request", uniprot_id)
            if i == 4:
                raise e
    if response.status_code != 200:
        raise ValueError(f"UniProt ID {uniprot_id} not found.")
    data = response.json()

    # Get protein name
    protein_name = (
        data.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value")
    )
    if not protein_name:
        submitted = data.get("proteinDescription", {}).get("submissionNames", [])
        protein_name = submitted[0]["fullName"]["value"] if submitted else "protein"

    # Get species name
    species_data = data.get("organism", {}).get("scientificName", "organism")
    species = species_data.lower()
    if "homo sapiens" in species:
        species = "human"
    elif "mus musculus" in species:
        species = "mouse"
    elif "rattus norvegicus" in species:
        species = "rat"
    else:
        species = species.split()[0]  # fallback: genus only

    return f"{species} {protein_name.lower()}"
