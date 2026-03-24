# Training Tools

## `MolRGen_train_reinvent_generator`

Train a REINVENT model with custom metadata for reward definition.

This endpoint starts the training asynchronously and returns immediately with a job_id. Use `MolRGen_get_training_status` to check the job status and progress.

**Arguments:**
- `output_dir`: Output directory for model checkpoints (default: "./results")
- `num_train_epochs`: Number of training epochs (default: 10)
- `eval_batch_size`: Batch size for evaluation (default: 64)
- `batch_size`: Batch size for training (default: 64)
- `sigma`: Sigma parameter for REINVENT (default: 0.1)
- `learning_rate`: Learning rate for REINVENT (default: 1e-5)
- `smiles_start`: Beginning of sequence to start with (default: ["\[s\]"])
- `metadata`: Metadata defining the reward function and objectives (GenerationVerifierInputMetadataModel with properties, objectives, and target values)

**Returns:**
Dict[str, Any]: Response including:
    - `status`: 'started' (training has begun asynchronously)
    - `job_id`: Unique identifier for the training job
    - `message`: Status message
    - `timestamp`: Start timestamp

**Example Usage:**
```python
from mol_gen_docking.reward.verifiers.generation_reward.input_metadata import GenerationVerifierInputMetadataModel

metadata = GenerationVerifierInputMetadataModel(
    properties=["QED", "SA"],
    objectives=["maximize", "minimize"],
    target=[0.8, 0.5]
)

params = ReinventTrainingParams(
    metadata=metadata,
    num_train_epochs=50,
    batch_size=64
)
result = train_reinvent_generator(params)
job_id = result["job_id"]

# Check training status later
status = get_training_status(job_id)
print(f"Training status: {status['status']}")
```

## `MolRGen_get_training_status`

Get the status of a REINVENT training job started with `MolRGen_train_reinvent_generator`.
If the status is running, it might take a few minutes to complete the job.

**Arguments:**
- `job_id`: The job identifier returned by `train_reinvent_generator`

**Returns:**
Dict[str, Any]: Job status information including:
    - `status`: 'started', 'running', 'completed', or 'failed'
    - `job_id`: The job identifier
    - `message`: Status message
    - `start_time`: When the job was started
    - `end_time`: When the job completed (if finished)
    - `output`: Training output (if available)
    - `command`: Command being executed (if running)

**Example Usage:**
```python
# Start training
result = train_reinvent_generator(params)
job_id = result["job_id"]

# Check status
status = get_training_status(job_id)
print(f"Status: {status['status']}")
if status['status'] == 'completed':
    print(f"Output: {status['output']}")
```
