"""
Launch SFT training and evaluation pipeline with SLURM job dependencies.
Configuration is read from a YAML file.
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SFTPipelineConfig:
    """Configuration handler for SFT training and evaluation pipeline."""

    def __init__(self, config_path: str, scratch_path: Optional[str] = None):
        """Load and parse YAML configuration file.

        Args:
            config_path: Path to YAML configuration file
            scratch_path: Optional path to prepend to dataset_path and output_path
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        self.model_name = self.config.get("model_name", "unk")

        self.scratch_path = scratch_path

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate required configuration fields."""
        required_keys = ["training", "evaluation"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration section: '{key}'")

        # Validate training config
        training_required = ["dataset_path", "config_train", "output_path"]
        for key in training_required:
            if key not in self.config["training"]:
                raise ValueError(f"Missing required training config: '{key}'")

        # Validate evaluation config
        evaluation_required = ["dataset_eval", "eval_config"]
        for key in evaluation_required:
            if key not in self.config["evaluation"]:
                raise ValueError(f"Missing required evaluation config: '{key}'")

    def _resolve_path(self, path: str) -> str:
        """Prepend scratch_path to a path if scratch_path is set.

        Args:
            path: The path to resolve

        Returns:
            The resolved path with scratch_path prepended if applicable
        """
        if self.scratch_path:
            # Don't prepend if path is already absolute and different from scratch_path
            if os.path.isabs(path):
                return os.path.join(self.scratch_path, os.path.basename(path))
            return os.path.join(self.scratch_path, path)
        return path

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration with resolved paths."""
        config = self.config.get("training", {})
        # Create a copy with resolved paths
        resolved_config = config.copy()
        resolved_config["dataset_path"] = self._resolve_path(config["dataset_path"])
        resolved_config["output_path"] = self._resolve_path(config["output_path"])
        return resolved_config  # type: ignore

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config.get("evaluation", {})  # type: ignore

    def _check_path_exists(self, path: str) -> None:
        """Check if a file or directory exists.

        Args:
            path: The path to check

        Raises:
            FileNotFoundError: If the file or directory does not exist
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

    def _count_dataset_lines(self, dataset_path: str) -> int:
        """Count the total number of conversation samples in the dataset file.

        Each line is a JSON object with a "conversations" key containing a list of conversation objects.
        This method sums up the lengths of all "conversations" lists across all lines.

        Args:
            dataset_path: Path to the JSONL dataset file

        Returns:
            The total number of conversation samples (sum of all conversation lengths)

        Raises:
            FileNotFoundError: If the dataset file does not exist
        """
        self._check_path_exists(dataset_path)

        total_samples = 0
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    obj = json.loads(line)
                    conversations = obj.get("conversations", [])
                    total_samples += len(conversations)
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse JSON line: {e}")
                    continue
        return total_samples

    def _count_dataset_total_tokens(
        self, dataset_path: str, ceil: Optional[int] = None
    ) -> int:
        """Count the total number of tokens in the dataset file.

        Each line is a JSON object with a "conversations" key containing a list of conversation objects.
        This method takes the heuristic of counting tokens as len(sequence) // 4 * 1.1, where sequence is the concatenated string of all conversation messages.
        The optional ceil parameter can be used to set a maximum token count (e.g., 4096) to avoid overestimating for very long conversations.
        Args:
            dataset_path: Path to the JSONL dataset file
            ceil: Optional maximum token count to apply to each conversation sequence
        Returns:
            The total number of tokens in the dataset (estimated)
        Raises:
            FileNotFoundError: If the dataset file does not exist
        """
        self._check_path_exists(dataset_path)

        total_tokens = 0
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    obj = json.loads(line)
                    conversations = obj.get("conversations", [])
                    for conv in conversations:
                        messages = conv.get("messages", [])
                        sequence = " ".join(msg.get("content", "") for msg in messages)
                        tokens = int(len(sequence) / 4 * 1.1)  # Heuristic token count
                        if ceil is not None:
                            tokens = min(tokens, ceil)
                        total_tokens += int(tokens)
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse JSON line: {e}")
                    continue
        return total_tokens

    def _load_sft_metadata(self, metadata_path: Optional[str] = None) -> Dict[str, Any]:
        """Load SFT metadata from JSON file.

        Args:
            metadata_path: Path to the SFT metadata JSON file. If None, uses default location.

        Returns:
            Dictionary containing SFT metadata

        Raises:
            FileNotFoundError: If the metadata file does not exist
        """
        if metadata_path is None:
            working_dir = os.path.expandvars("$HOME/MolGenDocking")
            metadata_path = os.path.join(
                working_dir,
                "slurm",
                "openrlhf",
                "configs",
                "sft_config",
                "sft_metadata.json",
            )

        metadata_path_ = Path(metadata_path)
        if not metadata_path_.exists():
            raise FileNotFoundError(f"SFT metadata file not found: {metadata_path_}")

        with open(metadata_path_) as f:
            metadata: Dict[str, Any] = json.load(f)
        return metadata

    def _calculate_training_time(
        self, total_training_steps: int, model_name: str, metadata: Dict[str, Any]
    ) -> tuple[int, int]:
        """Calculate training time in minutes and hours (rounded up).

        Args:
            total_training_steps: Total number of training steps
            model_name: Name of the model (e.g., 'Qwen3-4B')
            metadata: Dictionary containing SFT metadata with time_per_100steps

        Returns:
            Tuple of (training_time_minutes, estimated_hours_rounded_up)

        Raises:
            KeyError: If model_name not found in metadata or time_per_100steps not found
        """
        if model_name not in metadata:
            raise KeyError(f"Model '{model_name}' not found in SFT metadata")

        time_per_100steps = metadata[model_name].get("time_per_100steps")
        if time_per_100steps is None:
            raise KeyError(
                f"'time_per_100steps' not found for model '{model_name}' in metadata"
            )

        # Calculate training time in minutes
        # Formula: (total_steps / 100) * time_per_100steps
        training_time_minutes = (total_training_steps / 100) * time_per_100steps

        # Add 30 minutes buffer
        total_time_minutes = training_time_minutes + 30

        # Round up to the nearest hour
        estimated_hours = math.ceil(total_time_minutes / 60)

        logger.info(f"Training time calculation for model '{model_name}':")
        logger.info(f"  - Time per 100 steps: {time_per_100steps} minutes")
        logger.info(f"  - Estimated training time: {training_time_minutes:.2f} minutes")
        logger.info(f"  - With 30-minute buffer: {total_time_minutes:.2f} minutes")
        logger.info(f"  - Rounded up to hours: {estimated_hours} hours")

        return int(training_time_minutes), estimated_hours

    def _calculate_evaluation_time(
        self,
        total_training_steps: int,
        save_steps: int,
        model_name: str,
        metadata: Dict[str, Any],
    ) -> tuple[int, int]:
        """Calculate evaluation time in minutes and hours (rounded up).

        Args:
            total_training_steps: Total number of training steps
            save_steps: Number of steps between checkpoint saves
            model_name: Name of the model (e.g., 'Qwen3-4B')
            metadata: Dictionary containing SFT metadata with eval_time_per_ckpt

        Returns:
            Tuple of (evaluation_time_minutes, estimated_hours_rounded_up)

        Raises:
            KeyError: If model_name not found in metadata or eval_time_per_ckpt not found
        """
        if model_name not in metadata:
            raise KeyError(f"Model '{model_name}' not found in SFT metadata")

        eval_time_per_ckpt = metadata[model_name].get("eval_time_per_ckpt")
        if eval_time_per_ckpt is None:
            raise KeyError(
                f"'eval_time_per_ckpt' not found for model '{model_name}' in metadata"
            )

        # Calculate number of checkpoints
        # Formula: ceil(total_steps / save_steps)
        num_checkpoints = math.ceil(total_training_steps / save_steps)

        # Calculate evaluation time in minutes
        # Formula: num_checkpoints * eval_time_per_ckpt
        evaluation_time_minutes = num_checkpoints * eval_time_per_ckpt

        # Add 30 minutes buffer
        total_time_minutes = evaluation_time_minutes + 30

        # Round up to the nearest hour
        estimated_hours = math.ceil(total_time_minutes / 60)

        logger.info(f"Evaluation time calculation for model '{model_name}':")
        logger.info(f"  - Total training steps: {total_training_steps}")
        logger.info(f"  - Save steps interval: {save_steps}")
        logger.info(f"  - Number of checkpoints to evaluate: {num_checkpoints}")
        logger.info(f"  - Time per checkpoint: {eval_time_per_ckpt} minutes")
        logger.info(
            f"  - Estimated evaluation time: {evaluation_time_minutes} minutes ({num_checkpoints} ckpts × {eval_time_per_ckpt} min/ckpt)"
        )
        logger.info(f"  - With 30-minute buffer: {total_time_minutes} minutes")
        logger.info(f"  - Rounded up to hours: {estimated_hours} hours")

        return estimated_hours, num_checkpoints

    def calculate_eval_array_params(
        self,
        total_training_steps: int,
        save_steps: int,
        model_name: str,
        metadata: Dict[str, Any],
        target_hours: float = 3.0,
        buffer_minutes: int = 30,
    ) -> tuple[int, int, int]:
        """Calculate parameters for evaluation job array.

        Args:
            total_training_steps: Total number of training steps
            save_steps: Number of steps between checkpoint saves
            model_name: Name of the model
            metadata: Dictionary containing SFT metadata
            target_hours: Target duration for each job in hours
            buffer_minutes: Buffer time in minutes to subtract from target_hours

        Returns:
            Tuple of (num_jobs, checkpoints_per_job, job_time_hours)
        """
        if model_name not in metadata:
            raise KeyError(f"Model '{model_name}' not found in SFT metadata")

        eval_time_per_ckpt = metadata[model_name].get("eval_time_per_ckpt")
        if eval_time_per_ckpt is None:
            raise KeyError(
                f"'eval_time_per_ckpt' not found for model '{model_name}' in metadata"
            )

        # distinct checkpoints
        total_checkpoints = math.ceil(total_training_steps / save_steps)

        available_minutes = (target_hours * 60) - buffer_minutes
        if available_minutes <= 0:
            raise ValueError("Target hours too small for buffer")

        checkpoints_per_job = math.floor(available_minutes / eval_time_per_ckpt)
        if checkpoints_per_job < 1:
            checkpoints_per_job = 1
            # Recalculate time if one checkpoint takes longer than target
            job_time_minutes = eval_time_per_ckpt + buffer_minutes
            job_time_hours = math.ceil(job_time_minutes / 60)
        else:
            job_time_hours = int(math.ceil(target_hours))

        num_jobs = math.ceil(total_checkpoints / checkpoints_per_job)

        logger.info(f"Evaluation array calculation for '{model_name}':")
        logger.info(f"  - Total checkpoints: {total_checkpoints}")
        logger.info(f"  - Time per checkpoint: {eval_time_per_ckpt} min")
        logger.info(f"  - Target job time: {target_hours}h")
        logger.info(f"  - Checkpoints per job: {checkpoints_per_job}")
        logger.info(f"  - Number of jobs: {num_jobs}")
        logger.info(f"  - Job time limit: {job_time_hours}h")

        return num_jobs, checkpoints_per_job, job_time_hours


class SLURMJobLauncher:
    """Launcher for SLURM jobs."""

    def __init__(self, script_dir: Optional[str] = None, dry_run: bool = False):
        """
        Initialize the launcher.

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
        """
        Submit a SLURM job.

        Args:
            script_name: Name of the SLURM script (e.g., 'sft_model.sh')
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
            logger.warning("[DRY RUN] Would execute:" + " ".join(cmd))
            # Return a dummy job ID for dry run
            return "DRY_RUN_12345"

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Extract job ID from output (format: "Submitted batch job XXXXX")
            job_id = result.stdout.strip().split()[-1]
            return job_id
        except subprocess.CalledProcessError as e:
            logger.error(f"Error submitting job: {e.stderr}")
            raise

    def get_steps_per_epoch(
        self,
        dataset_path: str,
        batch_size: int,
        num_samples: int,
    ) -> int:
        """Calculate steps per epoch based on dataset and training configuration.

        Args:
            dataset_path: Path to the training dataset
            batch_size: Training batch size
            num_samples: Optional pre-counted number of samples in the dataset
        Returns:
            Steps per epoch
        """
        return (num_samples + batch_size - 1) // batch_size  # Ceiling division

    def launch_pipeline(self, config: SFTPipelineConfig) -> None:
        """
        Launch the SFT training and evaluation pipeline.

        Args:
            config: SFTPipelineConfig object with pipeline configuration

        Raises:
            FileNotFoundError: If dataset path does not exist
        """
        training_cfg = config.get_training_config()
        evaluation_cfg = config.get_evaluation_config()

        dataset_path = training_cfg["dataset_path"]
        output_path = training_cfg["output_path"]
        config_train_path = training_cfg["config_train"]

        # Check dataset path exists
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        logger.info(f"✓ Dataset path exists : {dataset_path}")

        # Always calculate steps to support array evaluation even if training is skipped
        # Count dataset samples (sum of all conversation lengths)
        num_samples = config._count_dataset_lines(dataset_path)

        # Load training config to get epochs and batch size
        with open(config_train_path) as f:
            train_config = yaml.safe_load(f)

        max_epochs = train_config.get("max_epochs", 1)
        train_batch_size = train_config.get("train_batch_size", 32)
        save_steps = train_config.get("save_steps", 50)
        max_len = train_config.get("max_len", 4096)
        packed = train_config.get("packing_samples", False)

        num_tokens = config._count_dataset_total_tokens(
            dataset_path, ceil=None if not packed else max_len
        )

        logger.info(f"Dataset samples: {num_samples} ({int(num_tokens / 1e6)}M tokens)")
        logger.info("Training configuration:")
        logger.info(f"  - max_epochs: {max_epochs}")
        logger.info(f"  - train_batch_size: {train_batch_size}")
        logger.info(f"  - save_steps: {save_steps}")
        logger.info(f"  - max_len: {max_len}")
        logger.info(f"  - packing_samples: {packed}")

        # Calculate training steps
        steps_per_epoch = self.get_steps_per_epoch(
            dataset_path,
            train_batch_size,
            num_samples,
        )
        total_training_steps = steps_per_epoch * max_epochs

        logger.info("Training step calculation:")
        logger.info(f"  - Steps per epoch: {steps_per_epoch}")
        logger.info(f"  - Total training steps: {total_training_steps}")

        # Check output path does not exist
        skip_training = False
        if Path(output_path).exists():
            logger.warning(
                f"Output path already exists: {output_path}. Skipping training and proceeding to evaluation."
            )
            skip_training = True
        else:
            logger.info("✓ Output path does not exist (ready to be created)")

        training_job_id = None
        training_time_minutes = None
        estimated_hours = None

        # Only train if output path doesn't exist
        if not skip_training:
            # Extract model name from config_train_path and load metadata
            model_name = config.model_name

            logger.info(f"Detected model name: {model_name}")

            # Load SFT metadata and calculate training time
            try:
                metadata = config._load_sft_metadata()
                training_time_minutes, estimated_hours = (
                    config._calculate_training_time(
                        total_training_steps, model_name, metadata
                    )
                )
            except (FileNotFoundError, KeyError) as e:
                logger.warning(f"Could not calculate training time: {e}")
                training_time_minutes = None
                estimated_hours = None

            # Prepare training job arguments
            training_args = [
                dataset_path,
                config_train_path,
                output_path,
            ]

            print("\nSubmitting SFT training job...")
            training_job_id = self.submit_job(
                "sft_model.sh", training_args, time_hours=estimated_hours
            )
            print(f"Training job submitted with ID: {training_job_id}")
        else:
            logger.info(
                "Skipping training job submission since output path already exists."
            )

        # Prepare evaluation job arguments
        ckpt_path = os.path.join(output_path, "ckpt")
        evaluation_args = [
            evaluation_cfg["dataset_eval"],
            evaluation_cfg["eval_config"],
            ckpt_path,
        ]

        # Calculate evaluation time / array params
        eval_estimated_hours = None
        num_checkpoints = None

        # New variables for array jobs
        num_jobs = 1
        checkpoints_per_job = None
        array_range = None

        if total_training_steps is not None and save_steps is not None:
            model_name = config.model_name
            try:
                metadata = config._load_sft_metadata()

                # Check if we should use array job (default: yes, with target 3h)
                try:
                    num_jobs, checkpoints_per_job, eval_estimated_hours = (
                        config.calculate_eval_array_params(
                            total_training_steps, save_steps, model_name, metadata
                        )
                    )

                    if num_jobs >= 1:
                        array_range = f"0-{num_jobs - 1}"
                        # Augment arguments for array job
                        evaluation_args.extend([save_steps, checkpoints_per_job])
                        logger.info(
                            f"Configuring as array job: range={array_range}, ckpts/job={checkpoints_per_job}"
                        )
                        num_checkpoints = math.ceil(total_training_steps / save_steps)

                except Exception as e:
                    logger.warning(
                        f"Failed to calculate array params, falling back to single job: {e}"
                    )
                    # Fallback to single huge job
                    eval_estimated_hours, num_checkpoints = (
                        config._calculate_evaluation_time(
                            total_training_steps, save_steps, model_name, metadata
                        )
                    )

            except (FileNotFoundError, KeyError) as e:
                logger.warning(f"Could not calculate evaluation time: {e}")
                eval_estimated_hours = None
                num_checkpoints = None

        print(
            "\nSubmitting SFT evaluation job"
            + (
                f" with dependency on training job {training_job_id}..."
                if training_job_id
                else "..."
            )
        )
        evaluation_job_id = self.submit_job(
            "sft_eval_model.sh",
            evaluation_args,
            dependencies=[training_job_id] if training_job_id else None,
            time_hours=eval_estimated_hours,
            array_range=array_range,
        )
        print(f"Evaluation job submitted with ID: {evaluation_job_id}")

        # Print summary
        print("\n" + "=" * 60)
        print("Job Summary:")
        print("=" * 60)
        if training_job_id:
            print(f"  Training Job ID:    {training_job_id}")
        else:
            print("  Training Job ID:    SKIPPED (output path already exists)")
        print(f"  Evaluation Job ID:  {evaluation_job_id}")
        if array_range:
            print(f"  Eval Array Range:   {array_range}")
            print(f"  Ckpts per Job:      {checkpoints_per_job}")
        print(f"  Checkpoint Path:    {ckpt_path}")
        if num_samples is not None:
            print(f"  Dataset Samples:    {num_samples}")
        if total_training_steps is not None:
            print(f"  Training Steps:     {total_training_steps}")
        if training_time_minutes is not None and estimated_hours is not None:
            print(f"  Est. Training Time: {training_time_minutes} minutes")
            print(f"  Est. Total Time:    {estimated_hours} hours (with 30min buffer)")
        if num_checkpoints is not None:
            print(f"  Total Checkpoints:  {num_checkpoints}")
        if eval_estimated_hours is not None:
            print(f"  Est. Eval Total:    {eval_estimated_hours} hours (per job)")
        print("=" * 60)
        print(
            "Evaluation will start"
            + (" after training completes." if training_job_id else ".")
        )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Launch SFT training and evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch pipeline with default configuration
  python launch_sft_pipeline.py config.yaml

  # Dry run (show commands without executing)
  python launch_sft_pipeline.py config.yaml --dry-run

  # With scratch path prepended to dataset and output paths
  python launch_sft_pipeline.py config.yaml --scratch-path /scratch/user
        """,
    )

    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them"
    )
    parser.add_argument(
        "--script-dir",
        type=str,
        default=None,
        help="Directory containing SLURM scripts (default: $HOME/MolGenDocking/slurm/openrlhf)",
    )
    parser.add_argument(
        "--scratch-path",
        type=str,
        default=None,
        help="Scratch path to prepend to dataset_path and output_path",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = SFTPipelineConfig(args.config, scratch_path=args.scratch_path)

        # Create launcher and launch pipeline
        launcher = SLURMJobLauncher(script_dir=args.script_dir, dry_run=args.dry_run)
        launcher.launch_pipeline(config)

        sys.exit(0)

    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
