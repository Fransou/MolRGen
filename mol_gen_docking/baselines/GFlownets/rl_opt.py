import json
import os
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import requests
import wandb
from datasets import load_from_disk
from rdkit import Chem
from tdc import Evaluator

from mol_gen_docking.baselines.GFlownets.args import get_config
from mol_gen_docking.baselines.GFlownets.trainer import MakeCustomTaskTrainer

N_REPEAT_TEST = 8
os.environ["WANDB_PROJECT"] = "REINVENT_HF-RL"


def get_reward_fn(
    metadata: Dict[str, Any], datasets_path: str, remote_rm_url: str
) -> Callable[[str | List[str]], float | List[float]]:
    response = requests.post(
        f"{remote_rm_url}/prepare_receptor",
        json={"metadata": [metadata]},
    )
    assert response.status_code == 200, response.text

    def reward_fn(completions: str | List[str], **kwargs: Any) -> float | List[float]:
        if isinstance(completions, str):
            completions = [completions]
        completions = [f"<answer> {s} </answer>" for s in completions]
        response = requests.post(
            f"{remote_rm_url}/get_reward",
            json={"query": completions, "metadata": [metadata] * len(completions)},
        )
        assert response.status_code == 200, response.text
        rewards: List[float] = response.json()["reward_list"]
        if isinstance(completions, str):
            return rewards[0]
        return rewards

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
            reward_fn = get_reward_fn(metadata, args.datasets_path, args.remote_rm_url)
            evaluator = EvalMolMetrics(reward_fn)
            wandb.init(
                project="GFlowNets-RL",
                config=config.__dict__,
            )
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
