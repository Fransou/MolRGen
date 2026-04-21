import json
import logging
import os
import random
import string
import subprocess
import time
from typing import Generator, List, Literal

import pandas as pd
import pytest
import requests

from molrgen.utils.property_utils import (
    CLASSICAL_PROPERTIES_NAMES,
)

# Define allowed accelerator types
AcceleratorType = Literal["cpu", "gpu"]
logger = logging.getLogger(__name__)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom pytest CLI options."""
    parser.addoption(
        "--data-path",
        action="store",
        default="data/molgendata",
        help="Path to the data directory.",
    )
    parser.addoption(
        "--accelerator",
        action="store",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Select the hardware accelerator to use for tests (cpu or gpu).",
    )
    parser.addoption(
        "--skip-docking",
        action="store_true",
        default=True,
        help="Skip docking tests when set.",
    )
    parser.addoption(
        "--docking",
        dest="skip_docking",
        action="store_false",
    )
    parser.addoption(
        "--start-server",
        action="store_true",
        default=False,
        help="Start the uvicorn server before running tests and stop it after.",
    )


# =============================================================================
# Data Fixtures
# =============================================================================

N_SMILES = 8
N_RANDOM_SMILES = 2
COMPLETIONS_PATTERN = [
    "Here is a molecule: <answer> \\boxed{{ {SMILES} }} </answer> what are its properties?",
    "This one looks interesting: COOC. Here is a molecule: <answer> \\boxed{{ {SMILES} }} </answer> \\boxed{{COOC}}",
    "Here is a {SMILES} molecule: <answer> \\boxed{{ {SMILES} }} </answer> what are {SMILES} its properties?",
    "[N] molecule:  <answer>{SMILES} what are its properties?",
    "[N] No SMILES here!",
]


@pytest.fixture(
    scope="session",
    params=[
        "minimize",
        "above 0.5",
        "below 0.5",
    ],
)  # type: ignore
def objective_to_test(request: pytest.FixtureRequest) -> str:
    """
    Pytest fixture returning a single objective to test.

    Example:
        def test_example(objective_to_test: str) -> None:
            ...
    """
    out: str = request.param
    return out


@pytest.fixture(scope="session")  # type: ignore
def data_path(request: pytest.FixtureRequest) -> str:
    """
    Pytest fixture returning the data path.

    Example:
        def test_example(data_path: str) -> None:
            ...
    """
    out: str = request.config.getoption("--data-path")
    return out


@pytest.fixture(scope="session")  # type: ignore
def properties_csv() -> pd.DataFrame:
    """
    Pytest fixture returning the properties CSV dataframe.

    Example:
        def test_example(properties_csv) -> None:
            ...
    """

    df = pd.read_csv("data/properties.csv", index_col=0)
    return df


@pytest.fixture(scope="session")  # type: ignore
def docking_targets(data_path: str) -> list[str]:
    """
    Pytest fixture returning the docking targets list.

    Example:
        def test_example(docking_targets: list[str]) -> None:
            ...
    """
    with open(os.path.join(data_path, "docking_targets.json")) as f:
        docking_targets_list: list = json.load(f)
    return docking_targets_list[:16]


@pytest.fixture(scope="session", params=list(range(N_RANDOM_SMILES + N_SMILES)))  # type: ignore
def idx_smiles(request: pytest.FixtureRequest) -> int:
    """
    Pytest fixture returning an index of a SMILES string.

    Example:
        def test_example(idx_smiles: int) -> None:
            ...
    """
    out: int = request.param
    return out


@pytest.fixture(scope="session")  # type: ignore
def prop(
    data_path: str,
    idx_smiles: int,
    docking_targets: List[str],
) -> str:
    with open(os.path.join(data_path, "names_mapping.json")) as f:
        properties_names_simple: dict = json.load(f)
    prop_list = [
        k
        for k in properties_names_simple.values()
        if k not in docking_targets and k in CLASSICAL_PROPERTIES_NAMES.values()
    ]
    random.shuffle(prop_list)
    out: str = prop_list[idx_smiles % len(prop_list)]
    return out


@pytest.fixture(scope="session")  # type: ignore
def smiles_list(properties_csv: pd.DataFrame) -> list[str]:
    """
    Pytest fixture returning a list of SMILES strings.

    Example:
        def test_example(smiles_list: list[str]) -> None:
            ...
    """
    characters = string.ascii_letters + string.digits
    full_list: list[str] = properties_csv["smiles"].sample(N_SMILES).tolist() + [
        "".join(random.choices(characters, k=random.randint(5, 15)))
        for _ in range(N_RANDOM_SMILES)
    ]
    # shuffle
    random.shuffle(full_list)
    return full_list


@pytest.fixture(scope="session", params=list(range(len(COMPLETIONS_PATTERN))))  # type: ignore
def completion_smile(
    smiles_list: list[str], idx_smiles: int, request: pytest.FixtureRequest
) -> tuple[str, str]:
    """
    Pytest fixture returning a single completion string with one SMILES.

    Example:
        def test_example(completions_list: list[str]) -> None:
            ...
    """
    smi = smiles_list[idx_smiles]
    pattern = COMPLETIONS_PATTERN[request.param]
    completion = pattern.replace("{SMILES}", smi)
    return completion, smi


@pytest.fixture(
    scope="session",
    params=[
        (n_smi, i_pattern)
        for n_smi in [2, 4]
        for i_pattern in range(len(COMPLETIONS_PATTERN))
    ],
)  # type: ignore
def completion_smiles(
    smiles_list: list[str], idx_smiles: int, request: pytest.FixtureRequest
) -> tuple[str, list[str]]:
    """
    Pytest fixture returning a single completion string with multiple SMILES.

    Example:
        def test_example(completions_list_multi_smiles: list[str]) -> None:
            ...
    """
    (n_smi, i_pattern) = request.param
    smi_chunk = smiles_list[idx_smiles : idx_smiles + n_smi]
    smi_joined = " ".join(smi_chunk)
    pattern = COMPLETIONS_PATTERN[i_pattern]
    completion = pattern.replace("{SMILES}", smi_joined)
    return completion, smi_chunk


@pytest.fixture(
    scope="session",
    params=[2, 4],
)  # type: ignore
def completions_smile(
    smiles_list: list[str], idx_smiles: int, request: pytest.FixtureRequest
) -> tuple[list[str], list[str]]:
    """
    Pytest fixture returning a single completion string with one SMILES.

    Example:
        def test_example(completions_list: list[str]) -> None:
            ...
    """
    n_pattern = request.param
    patterns = random.sample(COMPLETIONS_PATTERN, n_pattern)
    double_smi_list = smiles_list + smiles_list  # to avoid index error
    smis = double_smi_list[idx_smiles : idx_smiles + n_pattern]
    return [p.replace("{SMILES}", s) for p, s in zip(patterns, smis)], smis


@pytest.fixture(
    scope="session",
    params=[
        (n_pattern, n_smi)
        for n_pattern in [2, 4]
        for n_smi in [2, 8]  # 4 repetitions
    ],
)  # type: ignore
def completions_smiles(
    smiles_list: list[str], idx_smiles: int, request: pytest.FixtureRequest
) -> tuple[list[str], list[list[str]]]:
    """
    Pytest fixture returning a single completion string with multiple SMILES.

    Example:
        def test_example(completions_list_multi_smiles: list[str]) -> None:
            ...
    """
    (n_pattern, n_smi) = request.param
    patterns = random.sample(COMPLETIONS_PATTERN, n_pattern)
    double_smi_list = smiles_list + smiles_list  # to avoid index error
    smis_chunks = [
        double_smi_list[i : i + n_smi]
        for i in range(idx_smiles, idx_smiles + n_pattern * n_smi, n_smi)
    ]
    completions = [
        p.replace("{SMILES}", ", ".join(smi_chunk))
        for p, smi_chunk in zip(patterns, smis_chunks)
    ]
    return completions, smis_chunks


# =============================================================================
# Server Management
# =============================================================================

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5001
SERVER_STARTUP_TIMEOUT = 30  # seconds
SERVER_HEALTH_CHECK_INTERVAL = 5  # seconds


def _wait_for_server_ready(host: str, port: int, timeout: float) -> bool:
    """
    Wait for the server to become ready.

    Args:
        host: Server host address.
        port: Server port.
        timeout: Maximum time to wait in seconds.

    Returns:
        True if server is ready, False if timeout exceeded.
    """
    start_time = time.time()
    url = f"http://{host}:{port}/liveness"

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException as e:
            logger.info(f"Error, server probably not ready yet: {e}")
        time.sleep(SERVER_HEALTH_CHECK_INTERVAL)

    return False


def _is_server_running(host: str, port: int) -> bool:
    """
    Check if the server is already running.

    Args:
        host: Server host address.
        port: Server port.

    Returns:
        True if server is running, False otherwise.
    """
    try:
        response = requests.get(f"http://{host}:{port}/liveness", timeout=1)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@pytest.fixture(scope="session")  # type: ignore
def uvicorn_server(
    request: pytest.FixtureRequest,
) -> Generator[subprocess.Popen | None, None, None]:
    """
    Fixture to start and stop the uvicorn server.

    This fixture starts the uvicorn server before tests run and stops it after
    all tests are complete. It only starts the server if --start-server is passed.

    Yields:
        The subprocess.Popen object for the server, or None if not started.
    """
    start_server = request.config.getoption("--start-server")

    if not start_server:
        yield None
        return

    # Check if server is already running
    if _is_server_running(SERVER_HOST, SERVER_PORT):
        logger.info(f"\nServer already running on {SERVER_HOST}:{SERVER_PORT}")
        yield None
        return

    # Start the uvicorn server
    logger.info(f"\nStarting uvicorn server on {SERVER_HOST}:{SERVER_PORT}...")
    os.environ["buffer_time"] = "0"
    os.environ["data_path"] = "data/molgendata"  #
    process = subprocess.Popen(
        [
            "uvicorn",
            "--host",
            SERVER_HOST,
            "--port",
            str(SERVER_PORT),
            "molrgen.server:app",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    if _wait_for_server_ready(SERVER_HOST, SERVER_PORT, SERVER_STARTUP_TIMEOUT):
        logger.info(f"Server started successfully (PID: {process.pid})")
    else:
        process.terminate()
        process.wait()
        pytest.fail(f"Server failed to start within {SERVER_STARTUP_TIMEOUT} seconds")

    yield process

    # Cleanup: stop the server
    logger.info(f"\nStopping uvicorn server (PID: {process.pid})...")
    process.terminate()
    try:
        process.wait(timeout=10)
        logger.info("Server stopped successfully")
    except subprocess.TimeoutExpired:
        logger.info("Server did not stop gracefully, killing...")
        process.kill()
        process.wait()


@pytest.fixture(scope="session")  # type: ignore
def server_url(uvicorn_server: subprocess.Popen | None) -> str:
    """
    Fixture that returns the server URL.

    This fixture depends on uvicorn_server to ensure the server is running
    when --start-server is passed.

    Returns:
        The server URL string.
    """
    return f"http://{SERVER_HOST}:{SERVER_PORT}"


# =============================================================================
# Accelerator Fixtures
# =============================================================================


@pytest.fixture(scope="session")  # type: ignore
def accelerator(request: pytest.FixtureRequest) -> AcceleratorType:
    """
    Pytest fixture returning the selected accelerator.

    Example:
        def test_example(accelerator: AcceleratorType) -> None:
            if accelerator == "gpu":
                ...
    """
    accel: AcceleratorType = request.config.getoption("--accelerator")
    if bool(request.config.getoption("--skip-docking")):
        pytest.skip("Skipping docking tests")
    assert accel in ("cpu", "gpu")
    return accel


@pytest.fixture(scope="session")  # type: ignore
def has_gpu(accelerator: AcceleratorType) -> bool:
    """
    Convenience fixture: True if accelerator == 'gpu'.
    """
    return accelerator == "gpu"
