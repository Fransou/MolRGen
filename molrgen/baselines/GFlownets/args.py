import argparse
from typing import Any, Dict, Tuple

import torch
from gflownet.algo.config import (
    AlgoConfig,
    Backward,
    FMConfig,
    LossFN,
    NLoss,
    TBConfig,
    TBVariant,
)
from gflownet.config import Config, OptimizerConfig
from gflownet.data.config import ReplayConfig
from gflownet.models.config import ModelConfig
from gflownet.utils.config import ConditionalsConfig


def add_TB_config(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add Trajectory Balance configuration arguments to the parser"""
    group = parser.add_argument_group("Algorithm Config --- TB-Config")

    group.add_argument(
        "--tb_bootstrap_own_reward",
        action="store_true",
    )
    group.add_argument(
        "--tb_epsilon",
        type=float,
        default=None,
    )
    group.add_argument(
        "--tb_reward_loss_multiplier",
        type=float,
        default=1.0,
    )
    group.add_argument("--tb_variant", type=int, default=2)
    group.add_argument(
        "--tb_do_correct_idempotent",
        action="store_true",
    )
    group.add_argument(
        "--tb_do_parameterize_p_b",
        action="store_true",
    )
    group.add_argument(
        "--tb_do_predict_n",
        action="store_true",
    )
    group.add_argument(
        "--tb_do_sample_p_b",
        action="store_true",
    )
    group.add_argument(
        "--tb_do_length_normalize",
        action="store_true",
    )
    group.add_argument(
        "--tb_Z_learning_rate",
        type=float,
        default=1e-3,
    )
    group.add_argument("--tb_Z_lr_decay", type=float, default=50_000)
    group.add_argument(
        "--tb_loss_fn",
        type=int,
        default=0,
    )
    group.add_argument(
        "--tb_n_loss",
        type=int,
        default=0,
    )
    group.add_argument(
        "--tb_n_loss_multiplier",
        type=float,
        default=1.0,
    )
    group.add_argument(
        "--tb_backward_policy",
        type=int,
        default=1,
    )
    return group


def add_fm_config(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add Trajectory Balance configuration arguments to the parser"""
    group = parser.add_argument_group("Algorithm Config --- FM-Config")

    group.add_argument(
        "--fm_epsilon",
        type=float,
        default=1e-38,
    )
    group.add_argument(
        "--fm_balanced_loss",
        action="store_true",
    )
    group.add_argument(
        "--fm_leaf_coef",
        type=float,
        default=10,
    )
    group.add_argument(
        "--fm_correct_idempotent",
        action="store_true",
    )
    return group


def add_algo_config(
    parser: argparse.ArgumentParser,
) -> Dict[str, argparse._ArgumentGroup]:
    """Add algorithm configuration arguments to the parser"""
    group = parser.add_argument_group("Algorithm Config")

    group.add_argument(
        "--method",
        type=str,
        default="TB",
        help="The name of the algorithm to use (e.g. 'TB')",
    )

    group.add_argument(
        "--num_from_policy",
        type=int,
        default=64,
        help="The number of on-policy samples for a training batch",
    )
    group.add_argument(
        "--num_from_dataset",
        type=int,
        default=0,
        help="The number of dataset samples for a training batch",
    )
    group.add_argument(
        "--valid_num_from_policy",
        type=int,
        default=64,
        help="The number of on-policy samples for a validation batch",
    )

    group.add_argument(
        "--train_random_action_prob",
        type=float,
        default=0.1,
        help="The probability of taking a random action during training",
    )

    group.add_argument(
        "--train_det_after",
        type=int,
        default=None,
        help="Do not take random actions after this number of steps",
    )

    group.add_argument(
        "--sampling_tau",
        type=float,
        default=0.9,
        help="The EMA factor for the sampling model (theta_sampler = tau * theta_sampler + (1-tau) * theta)",
    )
    tb_group = add_TB_config(parser)
    fm_group = add_fm_config(parser)

    return {
        "main_algo": group,
        "tb_config": tb_group,
        "fm_config": fm_group,
    }


def add_opt_config(parser: argparse.ArgumentParser) -> None:
    """Add optimizer configuration arguments to the parser"""

    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="The learning rate"
    )

    parser.add_argument(
        "--lr_decay",
        type=float,
        default=10000,
        help="The learning rate decay (in steps, f = 2 ** (-steps / self.cfg.opt.lr_decay))",
    )


def add_replay_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use_replay_buffer",
        action="store_true",
    )
    parser.add_argument(
        "--replay_buffer_capacity",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--replay_buffer_warmup",
        type=int,
        default=100,
    )


def process_algo_args(
    args: argparse.Namespace, group_dict: Dict[str, argparse._ArgumentGroup]
) -> Dict[str, Dict[str, Any]]:
    # Get main algo args
    algo_args = {
        a.dest: getattr(args, a.dest) for a in group_dict["main_algo"]._group_actions
    }

    # Get tb args
    tb_args = {
        a.dest.replace("tb_", ""): getattr(args, a.dest)
        for a in group_dict["tb_config"]._group_actions
    }
    tb_args["variant"] = TBVariant(tb_args["variant"])
    tb_args["loss_fn"] = LossFN(tb_args["loss_fn"])
    tb_args["n_loss"] = NLoss(tb_args["n_loss"])
    tb_args["backward_policy"] = Backward(tb_args["backward_policy"])

    # Get fm args
    fm_args = {
        a.dest.replace("fm_", ""): getattr(args, a.dest)
        for a in group_dict["fm_config"]._group_actions
    }

    return {
        "algo": algo_args,
        "tb": tb_args,
        "fm": fm_args,
    }


def get_config() -> Tuple[Config, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dataset", type=str, default="jarod0411/zinc10M", help="Dataset name"
    )
    parser.add_argument(
        "--dataset", type=str, default="data/molgendata/eval_data/eval_prompts"
    )
    parser.add_argument("--datasets-path", type=str, default="data/molgendata")
    parser.add_argument(
        "--remote_rm_url",
        type=str,
        default="http://0.0.0.0:5001",
    )
    parser.add_argument(
        "--rewards_to_pick",
        type=str,  # Literal["docking_only", "std_only", "all"]
        default="all",
    )
    parser.add_argument(
        "--id_obj",
        type=int,
        default=0,
    )

    parser.add_argument("--output_dir", type=str, default="./results/GflowNets")
    parser.add_argument("--num_training_steps", type=int, default=10_000)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    # AlgoConfig
    group_dict = add_algo_config(parser)

    # ModelConfig TODO
    # OptimizerConfig
    add_opt_config(parser)

    # ReplayConfig
    add_replay_config(parser)

    # ConditionalsConfig

    args = parser.parse_args()

    algo_args = process_algo_args(args, group_dict)

    algo_config = AlgoConfig(
        tb=TBConfig(**algo_args["tb"]),
        fm=FMConfig(**algo_args["fm"]),
        **algo_args["algo"],
    )
    model_config = ModelConfig()

    opt_config = OptimizerConfig(
        learning_rate=args.learning_rate,
        lr_decay=args.lr_decay,
        clip_grad_param=1,
    )
    replay_config = ReplayConfig(
        use=args.use_replay_buffer,
        capacity=args.replay_buffer_capacity,
        warmup=args.replay_buffer_warmup,
    )
    cond_config = ConditionalsConfig()

    config = Config(
        log_dir=args.output_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_training_steps=args.num_training_steps,
        num_workers=args.num_workers,
        overwrite_existing_exp=True,
        print_every=1,
        validate_every=50,
        num_validation_gen_steps=4,
        pickle_mp_messages=True,
        algo=algo_config,
        model=model_config,
        opt=opt_config,
        replay=replay_config,
        cond=cond_config,
    )

    return config, args
