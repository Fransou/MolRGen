"""
Launch model evaluation given a dataset and model name.
Configuration is automatically retrieved from the config directory structure.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelEvaluationConfig:
    """Configuration handler for model evaluation."""

    def __init__(self, dataset: str, model_name: str):
        """Initialize and load model evaluation configuration.

        Args:
            dataset: Dataset name ('property_prediction' or 'molgendata')
            model_name: Model name without .yaml extension (e.g., 'gemma-3-27b-it')

        Raises:
            FileNotFoundError: If the config file does not exist
            ValueError: If dataset is not supported
        """
        if dataset not in ["property_prediction", "molgendata"]:
            raise ValueError(f"Unsupported dataset: {dataset}")

        self.dataset = dataset
        self.model_name = model_name

        # Construct config path
        working_dir = os.path.expandvars("$HOME/MolGenDocking")
        config_path = os.path.join(
            working_dir,
            "slurm",
            "openrlhf",
            "configs",
            dataset,
            f"{model_name}.yaml",
        )

        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Available models in {dataset}: {self._list_available_models(dataset)}"
            )

        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from: {config_path}")

    @staticmethod
    def _list_available_models(dataset: str) -> list:
        """List available model configs for a dataset.

        Args:
            dataset: Dataset name

        Returns:
            List of available model names
        """
        working_dir = os.path.expandvars("$HOME/MolGenDocking")
        config_dir = Path(working_dir) / "slurm" / "openrlhf" / "configs" / dataset

        if not config_dir.exists():
            return []

        return sorted([f.stem for f in config_dir.glob("*.yaml") if f.is_file()])

    def get_config(self) -> dict:
        """Get the full configuration dictionary.

        Returns:
            Configuration dictionary
        """
        return self.config  # type:ignore

    def get_config_path(self) -> str:
        """Get the absolute path to the config file.

        Returns:
            Path to config file
        """
        return str(self.config_path)


class SLURMEvaluationLauncher:
    """Launcher for SLURM evaluation jobs."""

    def __init__(self, script_dir: Optional[str] = None, dry_run: bool = False):
        """Initialize the launcher.

        Args:
            script_dir: Directory containing SLURM scripts. If None, uses default.
            dry_run: If True, print commands without executing them.
        """
        if script_dir is None:
            working_dir = os.path.expandvars("$HOME/MolGenDocking")
            script_dir = os.path.join(working_dir, "slurm", "openrlhf")

        self.script_dir = Path(script_dir)
        self.dry_run = dry_run

        if not self.script_dir.exists():
            raise FileNotFoundError(f"Script directory not found: {script_dir}")

    def submit_job(
        self,
        script_name: str,
        args: list,
        dependencies: Optional[list] = None,
        time_hours: Optional[int] = None,
        array_range: Optional[str] = None,
    ) -> str:
        """Submit a SLURM job.

        Args:
            script_name: Name of the SLURM script (e.g., 'evaluate_model.sh')
            args: List of arguments to pass to the script
            dependencies: List of job IDs this job depends on
            time_hours: Optional time limit in hours for the job
            array_range: Optional SLURM array range (e.g., '0-4')

        Returns:
            Job ID of the submitted job
        """
        script_path = self.script_dir / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        cmd = ["sbatch"]

        if dependencies:
            # Use afterany dependency type (job continues regardless of dependency success)
            dep_str = ":".join(str(dep_id) for dep_id in dependencies)
            cmd.extend(["--dependency", f"afterany:{dep_str}"])

        if time_hours is not None:
            # Format time as HH:00:00
            time_str = f"{time_hours:02d}:00:00"
            cmd.extend(["--time", time_str])
            logger.info(f"Setting job time limit to: {time_str}")

        if array_range is not None:
            cmd.extend(["--array", array_range])
            logger.info(f"Setting job array range to: {array_range}")

        cmd.append(str(script_path))
        cmd.extend(str(arg) for arg in args)

        if self.dry_run:
            logger.warning("[DRY RUN] Would execute: " + " ".join(cmd))
            # Return a dummy job ID for dry run
            return "DRY_RUN_12345"

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Extract job ID from output (format: "Submitted batch job XXXXX")
            job_id = result.stdout.strip().split()[-1]
            logger.info(f"Job submitted successfully with ID: {job_id}")
            return job_id
        except subprocess.CalledProcessError as e:
            logger.error(f"Error submitting job: {e.stderr}")
            raise

    def launch_evaluation(
        self,
        dataset: str,
        model_name: str,
        config: ModelEvaluationConfig,
        array_range: Optional[str] = None,
        time_hours: Optional[int] = None,
        dependencies: Optional[list] = None,
    ) -> str:
        """Launch model evaluation job.

        Args:
            dataset: Dataset name ('property_prediction' or 'molgendata')
            model_name: Model name
            config: ModelEvaluationConfig object
            array_range: Optional array range for job array submission (e.g., '0-4')
            time_hours: Optional time limit in hours
            dependencies: Optional list of job IDs to depend on

        Returns:
            Job ID of the submitted evaluation job
        """
        config_path = config.get_config_path()

        # Prepare evaluation job arguments
        evaluation_args = [dataset, config_path]

        logger.info(f"Launching evaluation for dataset: {dataset}, model: {model_name}")
        logger.info(f"Config path: {config_path}")

        evaluation_job_id = self.submit_job(
            "evaluate_model.sh",
            evaluation_args,
            dependencies=dependencies,
            time_hours=time_hours,
            array_range=array_range,
        )

        return evaluation_job_id


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Launch model evaluation for a given dataset and model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch evaluation for property_prediction with gemma-3-27b-it
  python launch_model_evaluation.py property_prediction gemma-3-27b-it

  # Launch evaluation for molgendata with Qwen3-30B_thinking
  python launch_model_evaluation.py molgendata Qwen3-30B_thinking

  # Launch with array job (0-4 array)
  python launch_model_evaluation.py property_prediction gemma-3-27b-it --array-range 0-4

  # Dry run (show command without executing)
  python launch_model_evaluation.py property_prediction gemma-3-27b-it --dry-run

  # With custom time limit (in hours)
  python launch_model_evaluation.py property_prediction gemma-3-27b-it --time-hours 6
        """,
    )

    parser.add_argument(
        "dataset",
        type=str,
        choices=["property_prediction", "molgendata"],
        help="Dataset name",
    )
    parser.add_argument(
        "model_name", type=str, help="Model name (config filename without .yaml)"
    )
    parser.add_argument(
        "--array-range",
        type=str,
        default=None,
        help="SLURM array range (e.g., '0-4')",
    )
    parser.add_argument(
        "--time-hours",
        type=int,
        default=None,
        help="Job time limit in hours",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--script-dir",
        type=str,
        default=None,
        help="Directory containing SLURM scripts (default: $HOME/MolGenDocking/slurm/openrlhf)",
    )
    parser.add_argument(
        "--dependency",
        type=str,
        nargs="+",
        default=None,
        help="Job IDs to depend on",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models for the specified dataset and exit",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        logger.info(
            f"Loading configuration for dataset={args.dataset}, model={args.model_name}"
        )
        config = ModelEvaluationConfig(args.dataset, args.model_name)

        # If --list-models is set, just list and exit
        if args.list_models:
            available = ModelEvaluationConfig._list_available_models(args.dataset)
            print(f"\nAvailable models for '{args.dataset}':")
            for model in available:
                marker = " [SELECTED]" if model == args.model_name else ""
                print(f"  - {model}{marker}")
            sys.exit(0)

        # Create launcher and launch evaluation
        launcher = SLURMEvaluationLauncher(
            script_dir=args.script_dir, dry_run=args.dry_run
        )

        print("\n" + "=" * 60)
        print("Model Evaluation Launch")
        print("=" * 60)
        print(f"  Dataset:     {args.dataset}")
        print(f"  Model:       {args.model_name}")
        print(f"  Config:      {config.get_config_path()}")
        if args.array_range:
            print(f"  Array Range: {args.array_range}")
        if args.time_hours:
            print(f"  Time Limit:  {args.time_hours} hours")
        if args.dependency:
            print(f"  Dependencies: {', '.join(args.dependency)}")
        if args.dry_run:
            print("  [DRY RUN MODE]")
        print("=" * 60 + "\n")

        evaluation_job_id = launcher.launch_evaluation(
            dataset=args.dataset,
            model_name=args.model_name,
            config=config,
            array_range=args.array_range,
            time_hours=args.time_hours,
            dependencies=args.dependency,
        )

        print(f"\nEvaluation job submitted with ID: {evaluation_job_id}")
        print("=" * 60)

        sys.exit(0)

    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
