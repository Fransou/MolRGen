import argparse
from pathlib import Path
from typing import Any

import jsonlines
import yaml
from tqdm import tqdm

from molrgen.data.meeko_process import ReceptorProcess
from molrgen.reward.molecular_verifier import (
    MolecularVerifier,
)
from molrgen.server_utils.server_setting import (
    MolecularVerifierServerSettings,
)

verifier_settings = MolecularVerifierServerSettings()
reward_scorer = MolecularVerifier(verifier_settings.to_molecular_verifier_config())
receptor_process: None | ReceptorProcess = None


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score molecular completions.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input file containing molecular completions.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--mol-generation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=None,
        help="Iteration number if used during inference.",
    )
    args = parser.parse_args()

    if args.input_file.endswith(".yaml"):
        # Config file from inference, override input_file with output_path in the config
        with open(args.input_file) as f:
            config = yaml.safe_load(f)
        args.input_file = config["output_path"]
        if not args.input_file.endswith(".jsonl"):
            args.input_file += ".jsonl"
        args.input_file = args.input_file.format(
            iter=args.iter if args.iter is not None else "final"
        )
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mol_generation:
        receptor_process = ReceptorProcess(
            data_path=verifier_settings.data_path, pre_process_receptors=True
        )

    if Path(args.input_file).is_file():
        input_files = [args.input_file]
    else:
        directory = Path(args.input_file)
        input_files = [
            str(f)
            for f in directory.glob("*.jsonl")
            if not str(f).endswith("_scored.jsonl")
            and not Path(str(f).replace(".jsonl", "_scored.jsonl")).exists()
        ]

    for i, input_file in enumerate(input_files):
        output_path = input_file.replace(".jsonl", "_scored.jsonl")
        print(f"[{i}/{len(input_files)}] === Computing Properties")
        completions: dict[str, list[dict[str, Any]]] = {}

        # Group completions by target property
        with jsonlines.open(input_file) as reader:
            for item in reader:
                if not args.mol_generation:
                    id = item["metadata"]["prompt_id"]
                    if id not in completions:
                        completions[id] = []
                    completions[id].append(item)
                else:
                    props = item["metadata"]["properties"]
                    found = False
                    assert receptor_process is not None
                    for p in props:
                        if p in receptor_process.pockets:
                            if p not in completions:
                                completions[p] = []
                            completions[p].append(item)
                            found = True
                            break
                    if not found:
                        if "no_docking" not in completions:
                            completions["no_docking"] = []
                        completions["no_docking"].append(item)
        completions_ordered: list[dict[str, Any]] = [
            item for sublist in completions.values() for item in sublist
        ]

        # Computing rewards
        all_responses: list[float] = []
        all_metas: list[dict[str, Any]] = []
        for idx in tqdm(
            range(0, len(completions_ordered), args.batch_size),
            desc="Scoring completions",
        ):
            batch = completions_ordered[
                idx : min(idx + args.batch_size, len(completions_ordered))
            ]
            # 1 pre-process
            if args.mol_generation:
                all_targets = []
                for item in batch:
                    all_targets.extend(item["metadata"]["properties"])
                all_targets = list(set(all_targets))

                assert receptor_process is not None
                all_targets = [r for r in all_targets if r in receptor_process.pockets]
                print(f"Processing targets:\n{all_targets}")
                receptor_process.process_receptors(
                    all_targets, allow_bad_res=True, use_pbar=False
                )
            # 2 get reward
            output = reward_scorer.get_score(
                completions=[item["output"] for item in batch],
                metadata=[item.get("metadata", {}) for item in batch],
            )
            all_responses.extend(output.rewards)
            all_metas.extend([m.model_dump() for m in output.verifier_metadatas])

        # Save results
        results = []
        for item, r, r_metadata in zip(completions_ordered, all_responses, all_metas):
            results.append(
                {
                    "output": item["output"],
                    "metadata": item.get("metadata", {}),
                    "reward": r,
                    "reward_meta": r_metadata,
                }
            )

        print("=== Saving")
        with jsonlines.open(output_path, mode="w") as writer:
            writer.write_all(tqdm(results, desc=f"Saving {output_path} |"))
