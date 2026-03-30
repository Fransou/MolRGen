---
name: molecular-toolkit
description: Tools for molecular generation, RDKit property calculation, and docking target analysis.
license: Apache-2.0
compatibility: Python 3.12+
user-invocable: true
mcp_servers:
  - MolRGen
allowed-tools:
  - read_file
  - grep
  - ask_user_question
  - write_file
---

# Molecular Toolkit

## When to use this skill
Use the `molecular_toolkit` skill when you need to perform tasks related to molecular generation, property calculation, or docking target identification. This skill provides a set of tools that can assist in various stages of molecular design and analysis.
Notably REINVENT training for molecular generation.

**Key Hyperparameters Used Throughout:**
- **Number of epochs** - Training duration
- **Learning rate** - Optimization step size
- **Batch size** - Number of molecules per group (larger batch sizes often improve results but increase computational cost)
- **Sigma** - Regularization term enforcing the model's distribution not to diverge from a prior, avoiding invalid SMILES generation
- **Smiles start** - Prefix list conditioning molecular generation (should always start with the token `<s>`)

## Molecular Generation Workflow
This workflow orchestrates multiple sub-agents across four stages to optimize molecular generation.
The process begins by identifying the corresponding objectives in the [properties](skills/molecular_toolkit/assets/properties.json) file (do not directly open it as it is a large file, but you can search for specific properties within it), to obtain the id of the property that can be given to the MolRGen mcp server.

### Stage 1: Sigma Parameter Exploration (Parallel Sub-agents)
Spawn `reinvent_trainer` sub-agents that explore the effect of the sigma regularization parameter on training quality:
- **Sub-agent 1a (σ=0.1)** - Conservative regularization
- **Sub-agent 1b (σ=1.0)** - Loose regularization

Each agent will return high scoring molecules they obtained.

### Stage 2: Hyperparameter Optimization & Structure Analysis (Sequential Sub-agents)

**Sub-agent 2a: Hyperparameter Aggregation** (`training_analyzer`): Collects results from Stage 1 sub-agents and provides recommendations for the best hyperparameter configuration.

**Sub-agent 2b: Structure Analysis & Prefix Suggestion** (`structure_analyzer`): Analyzes the top molecules from Stage 1 and provides prefixes that can be used to condition the REINVENT training.

### Stage 3: Final Training with Optimal Configuration (Parallel Sub-agents)

Spawn 3 `reinvent_trainer` sub-agents that train models with the optimal hyperparameters from Stage 2a, each using different conditioning strategies:

**Sub-agent 3a: Unconditional Generation**
- No SMILES conditioning
- Generates 128 molecules (eval_batch_size=128)

**Sub-agent 3b: Prefix-Conditioned Generation**
- Prefix conditioning from Stage 2b suggestions
- Generates 128 molecules (eval_batch_size=128)

**Sub-agent 3c: Multi-Prefix Generation**
- Combined conditioning: uses both suggested prefixes AND the neutral prefix `<s>`
- Generates 128 molecules (eval_batch_size=128)

### Stage 4: Final Analysis & Summary

Aggregate results from Stage 3 sub-agents:
1. Collect all 384 molecules (128 × 3 from each sub-agent)and select 128 high-scoring molecules (while ensuring diversity).
2. Generate comprehensive summary report (summary_results.md) containing:
   - An analysis of the hyperparameter exploration results from Stage 1
   - A summary of the structure analysis and prefix suggestions from Stage 2
   - Performance comparison of the three conditioning strategies
   - The 128 molecules kept and their scores
3. Finally display the best molecule directly to the user.
