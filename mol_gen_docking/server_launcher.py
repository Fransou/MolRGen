"""Server launcher that initializes Ray with placement groups and starts the FastAPI server.

This script:
1. Initializes a Ray server with 8 CPUs
2. Creates a placement group with 4 CPUs
3. Sets the placement group name in environment variables
4. Launches the FastAPI server with uvicorn
"""

import argparse
import logging
import os
import sys

import ray
from ray.util.placement_group import PlacementGroup, placement_group

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server_launcher")


def initialize_ray(num_cpus: int = 8, namespace: str = "molecular_verifier") -> None:
    """Initialize Ray with specified resources.

    Args:
        num_cpus: Number of CPUs to allocate for Ray
        namespace: Ray namespace to use
    """
    logger.info(f"Initializing Ray with {num_cpus} CPUs and namespace '{namespace}'")

    if ray.is_initialized():
        logger.warning("Ray is already initialized. Shutting down previous instance.")
        ray.shutdown()

    ray.init(num_cpus=num_cpus, namespace=namespace)
    logger.info("Ray initialized successfully")


def create_placement_group(
    num_cpus: int = 4, pg_name: str = "molecular_verifier_pg"
) -> PlacementGroup:
    """Create a placement group for the server.

    Args:
        num_cpus: Number of CPUs to allocate to the placement group
        pg_name: Name for the placement group

    Returns:
        PlacementGroup: The created placement group
    """
    logger.info(f"Creating placement group '{pg_name}' with {num_cpus} CPUs")

    pg: PlacementGroup = placement_group(
        [{"CPU": num_cpus}],
        strategy="STRICT_PACK",
        name=pg_name,
    )

    # Wait for the placement group to be ready
    ray.get(pg.ready())
    logger.info(f"Placement group '{pg_name}' created and ready")

    return pg


def launch_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    log_level: str = "info",
) -> None:
    """Launch the FastAPI server with uvicorn.

    Args:
        host: Host address to bind to
        port: Port to bind to
        workers: Number of uvicorn workers
        log_level: Logging level for uvicorn
    """
    import uvicorn

    logger.info(f"Launching server on {host}:{port} with {workers} worker(s)")
    # Add environment variables to ensure the server can access the placement group and namespace
    os.environ["PG_NAME"] = os.environ.get("PG_NAME", "molecular_verifier_pg")
    os.environ["RAY_NAMESPACE"] = os.environ.get("RAY_NAMESPACE", "molecular_verifier")

    uvicorn.run(
        "mol_gen_docking.server:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False,
    )


def main() -> None:
    """Main entry point for the server launcher."""
    parser = argparse.ArgumentParser(
        description="Launch the Molecular Verifier server with Ray placement groups"
    )
    parser.add_argument(
        "--ray-cpus",
        type=int,
        default=8,
        help="Number of CPUs to allocate for Ray (default: 8)",
    )
    parser.add_argument(
        "--pg-cpus",
        type=int,
        default=4,
        help="Number of CPUs for the placement group (default: 4)",
    )
    parser.add_argument(
        "--pg-name",
        type=str,
        default="molecular_verifier_pg",
        help="Name for the placement group (default: molecular_verifier_pg)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="molecular_verifier",
        help="Ray namespace (default: molecular_verifier)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to bind to (default: 5001)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn workers (default: 1)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Logging level (default: info)",
    )

    args = parser.parse_args()

    try:
        # Step 1: Initialize Ray
        initialize_ray(num_cpus=args.ray_cpus, namespace=args.namespace)

        # Step 2: Create placement group
        create_placement_group(num_cpus=args.pg_cpus, pg_name=args.pg_name)

        # Step 3: Set environment variables for the server to use
        os.environ["PG_NAME"] = args.pg_name
        os.environ["RAY_NAMESPACE"] = args.namespace

        logger.info(
            f"Environment variables set: PG_NAME={args.pg_name}, RAY_NAMESPACE={args.namespace}"
        )

        # Step 4: Launch the server
        launch_server(
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level,
        )

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during server launch: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if ray.is_initialized():
            logger.info("Shutting down Ray")
            ray.shutdown()


if __name__ == "__main__":
    main()
