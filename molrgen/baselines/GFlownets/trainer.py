from typing import Any, Callable, Dict, List, Tuple

import torch
from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.config import Config
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.models.bengio2021flow import FRAGMENTS
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as RDMol
from tdc import Evaluator
from torch import Tensor


class CustomTask(GFNTask):
    """A toy task where the reward is the number of rings in the molecule."""

    def __init__(
        self,
        reward_fn: Callable[[List[str] | str], float | List[float]],
        cfg: Config,
        *args: Any,
        **kwargs: Any,
    ):
        self.custom_reward_fn = reward_fn
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def sample_conditional_information(self, n: int, train_it: int) -> Any:
        return self.temperature_conditional.sample(n)

    def compute_obj_properties(
        self, mols: List[RDMol], *args: Any, **kwargs: Any
    ) -> Tuple[ObjectProperties, Tensor]:
        rs = torch.tensor(
            self.custom_reward_fn([Chem.MolToSmiles(m) for m in mols])
        ).float()

        return ObjectProperties(rs.reshape((-1, 1))), torch.ones(len(mols)).bool()

    def cond_info_to_logreward(
        self, cond_info: Dict[str, Tensor], obj_props: ObjectProperties
    ) -> LogScalar:
        scalar_logreward = torch.as_tensor(obj_props).squeeze().clamp(min=1e-8).log()
        return LogScalar(
            self.temperature_conditional.transform(cond_info, scalar_logreward)
        )


class MakeCustomTaskTrainer(StandardOnlineTrainer):
    def __init__(
        self,
        reward_fn: Callable[[List[str] | str], float | List[float]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.custom_reward_fn = reward_fn
        super().__init__(*args, **kwargs)

    def set_default_hps(self, cfg: Config) -> None:
        cfg.algo.illegal_action_logreward = -100

    def setup_task(self) -> None:
        # The task we created above
        self.task = CustomTask(self.custom_reward_fn, self.cfg)

    def setup_env_context(self) -> None:
        # The per-atom generation context
        self.ctx = FragMolBuildingEnvContext(
            max_frags=64,
            num_cond_dim=self.task.num_cond_dim,
            fragments=FRAGMENTS,
        )

    def setup(self) -> None:
        super().setup()
        self.valid_sampling_hooks.append(
            EvaluatorHook(["validity", "uniqueness", "diversity"], self.ctx)
        )


class EvaluatorHook:
    def __init__(self, fns: List[str], ctx: FragMolBuildingEnvContext):
        self.fns = [Evaluator(name) for name in fns]
        self.ctx = ctx

    def __call__(self, trajs: List[Any], *args: Any, **kwargs: Any) -> Dict[str, float]:
        valid_idcs = [i for i, traj in enumerate(trajs) if traj.get("is_valid", True)]

        mols = [
            self.ctx.graph_to_obj(tj["traj"][-1][0]) if i in valid_idcs else None
            for i, tj in enumerate(trajs)
        ]
        smiles = [
            Chem.MolToSmiles(mol) if mol is not None else "invalid"
            for i, mol in enumerate(mols)
        ]
        metrics = {fn.name: fn(smiles) for fn in self.fns}
        return metrics
