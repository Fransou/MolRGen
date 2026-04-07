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

## Molecular Generation Workflow
The process begins by identifying the corresponding objectives in the [properties](skills/molecular-toolkit/assets/properties.json) file (do not directly open it as it is a large file, but you can search for specific properties within it), to obtain the id of the property that can be given to the MolRGen mcp server.

Then start creating a plan to answer the question, and to orchestrate your sub-agents.

### Stage 1: HyperParameter Exploration

Spawn 2 `reinvent_trainer` sub-agents to explore different hyperparameter configurations.

**Key Hyperparameters Used Throughout:**
- **Number of epochs** - Training duration
- **Learning rate** - Optimization step size
- **Batch size** - Number of molecules per group (larger batch sizes often improve results but increase computational cost)
- **Sigma** - Regularization term enforcing the model's distribution not to diverge from a prior, avoiding invalid SMILES generation
Do not touch the eval-batch-size at this stage.

Analyze the results and repeat this process 2 times, adapting the hyperparameters based on the previous runs.

*If the objective is too hard to optimize, you can also change the optimization strategy, and compute the true reward a posteriori (e.g., by minimizing a property instead of reaching values below a threshold).*
### Stage 2: Hyperparameter Optimization & Structure Analysis

Spawn two parrallel sub-agents to analyze the results from Stage 1:

**Sub-agent 2a: Hyperparameter Analysis** (`training_analyzer`): Collects results from the Stage 1 sub-agents's training metrics and provides a comprehensive analysis to identify the optimal hyperparameter configuration.

**Sub-agent 2b: Structure Analysis & Prefix Suggestion** (`structure_analyzer`): Analyzes the top molecules from Stage 1 and identifies structural motifs and functional groups that are associated with high rewards.

### Stage 3: Final Analysis & Summary

Generate comprehensive summary report (summary_results.md) containing:
   - An analysis of the hyperparameter exploration results.
   - A summary of the structural analysis.
   - The best 128 molecules with their scores
