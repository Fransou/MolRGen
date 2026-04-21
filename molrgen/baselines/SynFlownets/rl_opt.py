import json
import os
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import wandb
from datasets import load_from_disk
from rdkit import Chem
from tdc import Evaluator

from molrgen.baselines.SynFlownets.args import get_config
from molrgen.baselines.SynFlownets.trainer import MakeCustomTaskTrainer
from molrgen.reward import (
    GenerationVerifierConfigModel,
    MolecularVerifier,
    MolecularVerifierConfigModel,
)

N_REPEAT_TEST = 8
os.environ["WANDB_PROJECT"] = "REINVENT_HF-RL"


def get_reward_fn(
    metadata: Dict[str, Any], datasets_path: str
) -> Callable[[str | List[str]], float | List[float]]:
    SCORER = MolecularVerifier(
        MolecularVerifierConfigModel(
            reward="property",
            parsing_method="none",
            generation_verifier_config=GenerationVerifierConfigModel(
                path_to_mappings=datasets_path
            ),
        )
    )

    def reward_fn(completions: str | List[str], **kwargs: Any) -> float | List[float]:
        if isinstance(completions, str):
            return SCORER([completions], metadata=[metadata], use_pbar=False).rewards[0]
        return SCORER(  # typing: ignore
            completions,
            metadata=[metadata] * len(completions),
            use_pbar=False,
        ).rewards

    return reward_fn


class EvalMolMetrics:
    def __init__(
        self,
        reward_fn: Callable[[str | List[str]], float | List[float]],
    ):
        self.reward_fn = reward_fn
        self.mol_evaluators = {
            name: Evaluator(name=name)
            for name in ["Validity", "Uniqueness", "Diversity"]
        }

    def __call__(self, mols: List[Chem.Mol]) -> Dict[str, float]:
        metrics_sub: Dict[str, List[float]] = {
            eval_name: [] for eval_name in self.mol_evaluators.keys()
        }
        metrics_sub["reward"] = []

        for i in range(0, len(mols), len(mols) // N_REPEAT_TEST):
            smiles = [Chem.MolToSmiles(mol) for mol in mols[i : i + N_REPEAT_TEST]]
            for eval_name in self.mol_evaluators:
                metrics_sub[eval_name].append(self.mol_evaluators[eval_name](smiles))
            metrics_sub["reward"].append(
                float(
                    np.mean(self.reward_fn(smiles)) * len(smiles) / n
                )  # Scale for non-generated smiles
            )

        metrics = {k: float(np.mean(m)) for k, m in metrics_sub.items()}
        return metrics


if __name__ == "__main__":
    config, args = get_config()
    dataset = load_from_disk(args.dataset)
    with open(os.path.join(args.datasets_path, "docking_targets.json")) as f:
        docking_targets = json.load(f)

    id = 0
    for row in dataset:
        metadata = {k: row[k] for k in ["properties", "objectives", "target"]}
        has_docking = any([prop in docking_targets for prop in metadata["properties"]])
        if args.rewards_to_pick == "std_only" and has_docking:
            continue
        elif args.rewards_to_pick == "docking_only" and not has_docking:
            continue

        print("=#=#=#=#" * 15)
        print("-#-#-#-#" * 5, f"[{id}] Task : {metadata}", "-#-#-#-#" * 5)
        print("=#=#=#=#" * 15)

        if args.id_obj == -1 or args.id_obj == id:
            reward_fn = get_reward_fn(metadata, args.datasets_path)
            evaluator = EvalMolMetrics(reward_fn)
            wandb.init(project="SynFlowNets-RL", allow_val_change=True)
            wandb.config.update({"prompt": metadata})  # type: ignore
            trial = MakeCustomTaskTrainer(reward_fn, config, print_config=False)
            trial.run()

            metrics = []
            for n in [1, 4, 16, 64, 256]:
                trajs = trial.algo.create_training_data_from_own_samples(trial.model, n)
                objs = [trial.ctx.graph_to_obj(i["result"]) for i in trajs]
                metric = evaluator(objs)
                metric["n"] = n
                metrics.append(metric)

            # Create Dataframe
            df = pd.DataFrame(metrics)
            print(df)
            table = wandb.Table(dataframe=df)
            wandb.log({"Final_evaluation": table})

        id += 1
