"""
Script to download and extract data from the SandboxAQ/SAIR dataset on Hugging Face from https://huggingface.co/datasets/SandboxAQ/SAIR.
"""

import argparse
import os
import subprocess

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm


def load_sair_parquet(destination_dir: str) -> None:
    """
    Downloads the sair.parquet file from the SandboxAQ/SAIR dataset and loads it
    into a pandas DataFrame.

    Args:
        destination_dir (str): The local path where the parquet file will be
                               downloaded. The directory will be created if it
                               doesn't exist.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the
                      sair.parquet file.
    """
    # --- 1. Setup and Repository Configuration ---
    repo_id = "SandboxAQ/SAIR"
    parquet_filename = "sair.parquet"

    print(f"Targeting repository: {repo_id}")
    print(f"Targeting file: {parquet_filename}")
    print(f"Destination directory: {destination_dir}")

    # Create the destination directory if it doesn't already exist
    os.makedirs(destination_dir, exist_ok=True)
    print("Ensured destination directory exists.")

    # --- 2. Download the Parquet file from the Hugging Face Hub ---
    download_path = os.path.join(destination_dir, parquet_filename)

    print(f"\nDownloading '{parquet_filename}'...")
    try:
        # Use hf_hub_download to get the file
        hf_hub_download(
            repo_id=repo_id,
            filename=parquet_filename,
            repo_type="dataset",
            local_dir=destination_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Successfully downloaded to '{download_path}'")
    except Exception as e:
        print(f"An error occurred while downloading '{parquet_filename}': {e}")
        return None


def download_and_extract_sair_structures(
    destination_dir: str, file_subset: list[str] | None = None, cleanup: bool = True
) -> None:
    """
    Downloads and extracts .tar.gz files from the SandboxAQ/SAIR dataset on Hugging Face.

    This function connects to the specified Hugging Face repository, identifies all
    .tar.gz files within the 'structures_compressed' directory, and downloads
    and extracts them to a local destination. It can download either all files
    or a specified subset.

    Args:
        destination_dir (str): The local path where the files will be downloaded
                               and extracted. The directory will be created if it
                               doesn't exist.
        file_subset (list[str], optional): A list of specific .tar.gz filenames
                                           to download. If None, all .tar.gz files
                                           in the directory will be downloaded.
                                           Defaults to None.
        cleanup (bool, optional): If True, the downloaded .tar.gz archive will be
                                  deleted after successful extraction. Defaults to True.

    Raises:
        ValueError: If any of the files specified in file_subset are not found
                    in the repository.
    """
    # --- 1. Setup and Repository Configuration ---
    repo_id = "SandboxAQ/SAIR"
    repo_folder = "structures_compressed"

    print(f"Targeting repository: {repo_id}")
    print(f"Destination directory: {destination_dir}")

    # Create the destination directory if it doesn't already exist
    os.makedirs(destination_dir, exist_ok=True)
    print("Ensured destination directory exists.")

    # --- 2. Get the list of relevant files from the Hugging Face Hub ---
    if os.path.exists("SAIR_repo_tars.json"):
        import json

        with open("SAIR_repo_tars.json") as f:
            repo_tars = json.load(f)
        print(f"Loaded {len(repo_tars)} .tar.gz filenames from cached JSON.")
    else:
        try:
            all_files = list_repo_files(repo_id, repo_type="dataset")
            # Filter for files within the specified folder that are tar.gz archives
            repo_tars = [
                f.split("/")[-1]
                for f in all_files
                if f.startswith(repo_folder + "/") and f.endswith(".tar.gz")
            ]
            print(f"Found {len(repo_tars)} total .tar.gz files in '{repo_folder}'.")
        except Exception as e:
            print(
                f"Error: Could not list files from repository '{repo_id}'. Please check the name and your connection."
            )
            print(f"Details: {e}")
            return

    # --- 3. Determine which files to download ---
    if file_subset:
        if (
            len(file_subset) == 2
            and isinstance(file_subset[0], int)
            and isinstance(file_subset[1], int)
        ):
            # Interpret as a range if two integers are provided
            start, end = file_subset
            file_subset = repo_tars[start:end]
            print(
                f"Interpreted subset as range: indices {start} to {end}, total {len(file_subset)} files."
            )

        # Validate that all requested files actually exist in the repository
        invalid_files = set(file_subset) - set(repo_tars)
        if invalid_files:
            raise ValueError(
                f"The following requested files were not found in the repository: {list(invalid_files)}"
            )

        files_to_download = file_subset
        print(f"A subset of {len(files_to_download)} files was specified for download.")
    else:
        files_to_download = repo_tars
        print("No subset specified. All .tar.gz files will be downloaded.")

    # --- 4. Download and Extract each file ---
    # Check if the files are already downloaded and extracted if only one requested
    structure_dir = os.path.join(destination_dir, "structures")
    if os.path.exists(structure_dir) and len(files_to_download) == 1:
        existing_files = os.listdir(structure_dir)
        if len(existing_files) == 50000:
            print(
                f"Directory '{structure_dir}' already exists and contains 50000 files. Skipping download."
            )
            return

    for filename in tqdm(files_to_download, desc="Processing files"):
        # Construct the full path within the repository
        repo_filepath = f"{repo_folder}/{filename}"

        download_path = os.path.join(destination_dir, repo_filepath)

        print(f"\nDownloading '{filename}'...")
        try:
            # Download the file from the Hub
            if not os.path.exists(os.path.join(destination_dir, repo_filepath)):
                hf_hub_download(
                    repo_id=repo_id,
                    filename=repo_filepath,
                    repo_type="dataset",
                    local_dir=destination_dir,
                    local_dir_use_symlinks=False,
                )
                print(f"Successfully downloaded to '{download_path}'")
            #
            # Extract the downloaded .tar.gz file
            print(f"Extracting '{filename}'...")
            subprocess.check_call(
                [
                    "tar",
                    "-xf",
                    download_path,
                    "-C",
                    destination_dir,
                ]
            )
            print(f"Successfully extracted contents to '{destination_dir}'")

        except Exception as e:
            print(f"An error occurred while processing '{filename}': {e}")
            continue

        finally:
            #     Clean up the downloaded archive if the flag is set and the file exists
            if cleanup and os.path.exists(download_path):
                os.remove(download_path)
                print(f"Cleaned up (deleted) '{download_path}'")

    print("\nOperation completed.")


if __name__ == "__main__":
    # --- Download the parquet dataset ---
    parser = argparse.ArgumentParser(
        description="Download and extract data from the SandboxAQ/SAIR dataset on Hugging Face."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sair_data",
        help="Directory to save downloaded data.",
    )
    parser.add_argument(
        "--download-parquet",
        action="store_true",
        help="Flag to download the sair.parquet file.",
        default=False,
    )
    parser.add_argument(
        "--start-subset",
        type=int,
        default=0,
        help="Start index for subset of structure files to download (inclusive).",
    )
    parser.add_argument(
        "--end-subset",
        type=int,
        default=1,
        help="End index for subset of structure files to download (exclusive).",
    )

    args = parser.parse_args()

    # Define a destination for the data
    output_directory = args.output_dir

    # Call the function to download and load the data
    if args.download_parquet:
        load_sair_parquet(destination_dir=output_directory)
    # --- Download a specific subset of structure tarballs ---
    # Define the specific files you want to download
    # Replace this with None to download *all* structures
    # (remember, this is >100 files of ~10GB each!)
    subset_to_get = [args.start_subset, args.end_subset]
    download_and_extract_sair_structures(
        destination_dir=output_directory, file_subset=subset_to_get
    )
