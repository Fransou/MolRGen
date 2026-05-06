#!/bin/bash
#SBATCH --job-name=orz_mol
#SBATCH --account=def-ibenayed
#SBATCH --time=72:00:00
#SBATCH --gpus=h100:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source $HOME/.bashrc
export WORKING_DIR=$HOME/MolGenDocking
export DATASET=molgendata

cp $SCRATCH/MolGenData/$DATASET.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xzf $DATASET.tar.gz

cd $WORKING_DIR
cp data/properties.csv $SLURM_TMPDIR

export DATA_PATH=$SLURM_TMPDIR/$DATASET
source $HOME/OpenRLHF/bin/activate
port=6379

ray start --head --node-ip-address 0.0.0.0

export docking_oracle=pyscreener
export scorer_exhaustiveness=4
uvicorn --host 0.0.0.0 --port 5001 molrgen.server:app &
sleep 15

wandb offline
export GPUS_PER_NODES=1
export PRETRAIN=$SCRATCH/ether0

#export DEBUG_MODE=1
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"setup_commands": ["wandb offline"]}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node $GPUS_PER_NODES \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node $GPUS_PER_NODES \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node $GPUS_PER_NODES \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node $GPUS_PER_NODES \
   --vllm_num_engines $GPUS_PER_NODES \
   --vllm_tensor_parallel_size 1 \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.8 \
   --pretrain $PRETRAIN \
   --remote_rm_url http://localhost:5000/get_reward \
   --save_path $SCRATCH/DockGen-4B \
   --ckpt_path $SCRATCH/checkpoint/DockGen-4B \
   --max_ckpt_num 5 \
   --save_steps 3 \
   --micro_train_batch_size 2 \
   --train_batch_size 8 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 8 \
   --n_samples_per_prompt 64 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 512 \
   --generate_max_len 4096 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.1 \
   --advantage_estimator reinforce \
   --prompt_data $SLURM_TMPDIR/$DATASET/train_prompts \
   --input_key prompt \
   --label_key metadata \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --enforce_eager \
   --use_wandb 95190474fa39dc888a012cd12b18ab9b094697ad
