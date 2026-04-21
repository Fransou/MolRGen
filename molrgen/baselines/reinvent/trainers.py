import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import torch
import torch.utils.data
from accelerate.utils import gather_object
from datasets import Dataset
from rdkit import Chem
from scipy.spatial.distance import squareform
from torch.utils.data import Sampler
from transformers import AutoTokenizer, EvalPrediction
from trl import GRPOTrainer
from trl.trainer.utils import RepeatSampler, nanmax, nanmin, nanstd, pad

from molrgen.evaluation.fingeprints_utils import get_sim_matrix

N_REPEAT_TEST = 8

logger = logging.getLogger(__name__)


def get_reward_fn(
    metadata: Dict[str, Any], datasets_path: str, remote_rm_url: str
) -> Callable[[str | List[str]], float | List[float]]:
    response = requests.post(
        f"{remote_rm_url}/prepare_receptor",
        json={"metadata": [metadata], "query": [""]},
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
        rewards: List[float] = response.json().get("rewards", [0.0] * len(completions))
        if isinstance(completions, str):
            return rewards[0]
        return rewards

    return reward_fn


class MolGenerationEvaluator:
    def __init__(self, qsim: float = 0.05):
        """
        Evaluator class measuring the validity, uniqueness and diversity of a set of smiles.
        Args:
            qsim: Quantile taken for the diversity measure
        """
        self.qsim = qsim
        self.keys = [
            "Validity",
            "Uniqueness",
            "Diversity",
        ]

    def __call__(self, smiles: List[str]) -> Dict[str, float]:
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        valid_mols = [m for m in mols if m is not None]

        out: Dict[str, float] = {}
        out["Validity"] = len(valid_mols) / len(mols) if len(mols) > 0 else 0.0
        if len(valid_mols) < 2:
            out["Uniqueness"] = 0.0
            out["Diversity"] = 0.0
        else:
            sim_mat = squareform(get_sim_matrix(mols=valid_mols))
            M = np.tri(sim_mat.shape[0], sim_mat.shape[0], k=-1, dtype=int).T
            out["Uniqueness"] = (
                (((sim_mat == 1.0).astype(int)) @ M).diagonal() == 0
            ).mean()
            out["Diversity"] = 1 - np.quantile(sim_mat, q=1 - self.qsim, axis=1).mean()

        return out


class EvalMolMetrics:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        reward_fn: Callable[[str | List[str]], float | List[float]],
    ):
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.mol_evaluator = MolGenerationEvaluator()

    def __call__(self, preds: EvalPrediction) -> Dict[str, float]:
        metrics_sub: Dict[str, List[float]] = {
            eval_name: [] for eval_name in self.mol_evaluator.keys
        }
        metrics_sub["reward"] = []
        n = len(preds.label_ids)
        steps = n // N_REPEAT_TEST
        for i in range(0, n, steps):
            sub_label_ids = preds.label_ids[i : i + steps]
            sub_label_ids[sub_label_ids == -100] = self.tokenizer.pad_token_id
            completions_text = self.tokenizer.batch_decode(
                sub_label_ids, skip_special_tokens=True
            )
            mol_eval_out = self.mol_evaluator(completions_text)
            for eval_name in mol_eval_out:
                metrics_sub[eval_name].append(mol_eval_out[eval_name])

            # for the reward, we remove duplicates and keep the top-n after this processing
            mols = [Chem.MolFromSmiles(smi) for smi in completions_text]
            smiles = []
            for mol in mols:
                if mol is not None:
                    smi = Chem.MolToSmiles(mol)
                    if smi not in smiles:
                        smiles.append(smi)
            reward = self.reward_fn(smiles)
            if reward == []:
                reward = [0.0]
            metrics_sub["reward"].append(
                float(
                    np.mean(reward) * len(smiles) / len(completions_text)
                )  # Scale for non-generated smiles
            )

        metrics = {k: float(np.mean(m)) for k, m in metrics_sub.items()}
        return metrics


class ReinventGRPOTrainer(GRPOTrainer):
    def __init__(
        self, compute_metrics: Any, n_repeat_test: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mol_evaluator = MolGenerationEvaluator()
        self.compute_metrics = compute_metrics
        self.n_repeat_test = n_repeat_test
        self.training_num_generations: int = copy.deepcopy(self.num_generations)

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Sampler:
        self.generation_config.num_return_sequences = 1
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=1,
            seed=self.args.seed,
        )

    def prediction_step(
        self,
        model: Any,
        inputs: Any,
        prediction_loss_only: Any,
        ignore_keys: Optional[list[str]] = None,
    ) -> Tuple[Any, Any, Any]:
        inputs = self._prepare_inputs(inputs)
        return torch.tensor(0.0), inputs["completion_ids"], inputs["completion_ids"]

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "train" if self.model.training else "eval"
        self.num_generations = self.training_num_generations if mode == "train" else 1
        outputs: dict[str, Union[torch.Tensor, Any]] = (
            super()._generate_and_score_completions(inputs)
        )

        completions_text = list(self._logs["completion"])  # Trick get all completions
        ### MOLECULE SPECIFIC METRICS ###
        mol_eval_out = self.mol_evaluator(completions_text)
        for eval_name in mol_eval_out:
            self._metrics[mode][eval_name].append(mol_eval_out[eval_name])
        self.num_generations = self.training_num_generations
        return outputs


class VanillaReinventTrainer(ReinventGRPOTrainer):
    def __init__(
        self, compute_metrics: Any, n_repeat_test: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(compute_metrics, n_repeat_test, *args, **kwargs)
        self.mol_evaluator = MolGenerationEvaluator()
        self.compute_metrics = compute_metrics
        self.n_repeat_test = n_repeat_test

    def _compute_loss(self, model: Any, inputs: Dict[str, Any]) -> Any:
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        # Get the reward
        reward = inputs["reward"].float()
        # Compute the Diff
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_diff = ref_per_token_logps - per_token_logps
        diff = per_token_diff.sum(-1)

        loss = (diff + self.beta * reward) ** 2
        return loss.mean()

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else 1
        prompts = [x["prompt"] for x in inputs]
        if mode == "eval":
            assert len(prompts) % self.generation_config.num_return_sequences == 0
            n_prompts_to_keep = (
                len(prompts) // self.generation_config.num_return_sequences
            )
            prompts = prompts[:n_prompts_to_keep]
        images = None

        (
            prompt_ids_list,
            completion_ids_list,
            num_items_in_batch,
            sampling_per_token_logps_list,
            forward_kwargs,
        ) = self._generate(prompts, images)

        # Convert lists of token IDs to padded tensors
        if mode == "train":
            prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        else:
            prompt_ids = [
                torch.tensor(ids, device=device)
                for ids in prompt_ids_list
                for _ in range(self.generation_config.num_return_sequences)
            ]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(
            prompt_ids, padding_value=self.pad_token_id, padding_side="left"
        )
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [
            torch.tensor(ids, device=device) for ids in completion_ids_list
        ]
        completion_mask = [
            torch.ones_like(ids, dtype=torch.long) for ids in completion_ids
        ]
        completion_ids = pad(
            completion_ids, padding_value=self.pad_token_id, padding_side="right"
        )
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [
                torch.tensor(logps, device=device)
                for logps in sampling_per_token_logps_list
            ]
            sampling_per_token_logps = pad(
                sampling_per_token_logps, padding_value=0.0, padding_side="right"
            )
        else:
            sampling_per_token_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor(
                [ids[-1] not in eos_and_pad for ids in completion_ids_list],
                device=device,
            )
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat(
            [prompt_ids, completion_ids], dim=1
        )  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)],
                dim=1,
            )

        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        num_images = None

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = (
                self.args.steps_per_generation * self.num_iterations
            )  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(
                    old_per_token_logps - sampling_per_token_logps
                )
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,
                        # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = (
                            self._get_per_token_logps_and_entropies(
                                self.model,
                                prompt_completion_ids,
                                attention_mask,
                                logits_to_keep,
                                batch_size=batch_size,
                                num_images=num_images,
                                **forward_kwargs,
                                # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                            )
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(
            prompt_ids, skip_special_tokens=True
        )
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        completions = completions_text
        ### MOLECULE SPECIFIC METRICS ###
        mol_eval_out = self.mol_evaluator(completions_text)
        for eval_name in mol_eval_out:
            self._metrics[mode][eval_name].append(mol_eval_out[eval_name])
        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(
            inputs,
            [
                p
                for p in prompts
                for _ in range(self.generation_config.num_return_sequences)
            ],
            completions,
            completion_ids_list,
        )

        # Apply weights to each reward function's output and sum
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            std_rewards = rewards.view(-1, num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
        elif self.scale_rewards == "batch":
            # Compute global std
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = (
            advantages.clone()
        )  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(
                std_func_rewards
            )
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(
            is_std_zero.float().mean().item()
        )

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = (
                torch.mean(delta)
                if delta.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            max_delta = (
                torch.max(delta)
                if delta.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
            "reward": rewards,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps

        return output
