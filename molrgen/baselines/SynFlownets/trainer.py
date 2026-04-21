import pickle
from typing import Any, Callable, Dict, List

from rdkit import Chem
from synflownet.algo.reaction_sampling import SynthesisSampler
from synflownet.config import Config
from synflownet.envs.synthesis_building_env import (
    ReactionTemplateEnv,
    ReactionTemplateEnvContext,
)
from synflownet.online_trainer import StandardOnlineTrainer
from tdc import Evaluator

from molrgen.baselines.GFlownets.trainer import CustomTask

ATOMS: list[str] = [
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "B",
    "Sn",
    "Ca",
    "Na",
    "Ba",
    "Zn",
    "Rh",
    "Ag",
    "Li",
    "Yb",
    "K",
    "Fe",
    "Cs",
    "Bi",
    "Pd",
    "Cu",
    "Si",
    "Ni",
    "As",
    "Cd",
    "H",
    "Hg",
    "Mg",
    "Mn",
    "Se",
    "Pt",
    "Sb",
    "Pb",
]


class MakeCustomTaskTrainer(StandardOnlineTrainer):
    def __init__(
        self,
        reward_fn: Callable[[List[str] | str], float | List[float]],
        config: Any,
        syn_flow_repo_root: str = ".",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.custom_reward_fn = reward_fn
        self.syn_flow_repo_root = syn_flow_repo_root
        super().__init__(config, *args, **kwargs)

    def set_default_hps(self, cfg: Config) -> None:
        cfg.algo.illegal_action_logreward = -100

    def setup_task(self) -> None:
        # The task we created above
        self.task = CustomTask(self.custom_reward_fn, self.cfg)

    def setup_env_context(self) -> None:
        # Load building blocks
        building_blocks_path = self.cfg.task.reactions_task.building_blocks_filename
        with open(building_blocks_path) as file:
            building_blocks = file.read().splitlines()

        # Load templates
        templates_path = self.cfg.task.reactions_task.templates_filename
        with open(templates_path) as file:
            reaction_templates = file.read().splitlines()

        precomputed_bb_masks_path = (
            self.cfg.task.reactions_task.precomputed_bb_masks_filename
        )

        with open(precomputed_bb_masks_path, "rb") as f:
            precomputed_bb_masks = pickle.load(f)
        print(self.cfg)
        self.ctx = ReactionTemplateEnvContext(
            atoms=ATOMS,
            num_cond_dim=self.task.num_cond_dim,
            building_blocks=building_blocks,
            reaction_templates=reaction_templates,
            precomputed_bb_masks=precomputed_bb_masks,
            fp_type=self.cfg.model.graph_transformer.fingerprint_type,
            fp_path=self.cfg.model.graph_transformer.fingerprint_path,
            strict_bck_masking=self.cfg.algo.strict_bck_masking,
        )
        self.env = ReactionTemplateEnv(
            reaction_templates=reaction_templates,
            building_blocks=building_blocks,
            precomputed_bb_masks=precomputed_bb_masks,
            fp_type=self.cfg.model.graph_transformer.fingerprint_type,
            fp_path=self.cfg.model.graph_transformer.fingerprint_path,
        )

    def setup(self) -> None:
        super().setup()
        self.valid_sampling_hooks.append(
            EvaluatorHook(["validity", "uniqueness", "diversity"], self.ctx)
        )

    def setup_sampler(self) -> None:
        self.sampler = SynthesisSampler(
            cfg=self.cfg,
            ctx=self.ctx,
            env=self.env,
            max_len=self.cfg.algo.max_len,
            correct_idempotent=self.cfg.algo.tb.do_correct_idempotent,
            pad_with_terminal_state=self.cfg.algo.tb.do_parameterize_p_b,
        )


class EvaluatorHook:
    def __init__(self, fns: List[str], ctx: ReactionTemplateEnvContext):
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
