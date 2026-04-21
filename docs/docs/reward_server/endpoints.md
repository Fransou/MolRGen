# Reward Server API Documentation

The Reward Server provides REST endpoints for molecular scoring and docking calculations. It uses FastAPI to handle asynchronous requests and Ray for distributed computation.

## Endpoints

### 1. Liveness Check

**Endpoint:** `GET /liveness`

**Description:** Health check endpoint to verify the server is running.

---

### 2. Get Reward

**Endpoint:** `POST /get_reward`

**Description:** Calculate molecular rewards (docking scores, validity, etc.) for generated molecules. This endpoint handles batched requests efficiently through a buffering system that groups concurrent requests before computation.

---

### 3. Prepare Receptor

**Endpoint:** `POST /prepare_receptor`

**Description:** Prepare protein receptors for docking calculations. This is an internal endpoint that preprocesses receptor files needed for AutoDock GPU computations.

**Request Body:**

```json
{
  "metadata": [
    {
      "properties": ["1a28", "2x5y"],
      "objectives": [],
      "target": []
    }
  ],
  "query": ["dummy"],
  "prompts": null
}
```

**Response (Success):**

```json
{
  "status": "Success"
}
```

**Response (No Preparation Needed):**

```json
{
  "status": "No need to prepare receptors for the selected docking oracle."
}
```

**Response (Error):**

```json
{
  "status": "Error",
  "info": "Receptors [1a28, 2x5y] could not be processed."
}
```

!!!note
    This endpoint is automatically called before reward calculation when using the AutoDock GPU oracle. It validates and preprocesses PDB receptor files. This results in larger time overhead during the first docking request for new receptors, but subsequent requests will be faster.

---


## Usage Example

```python
import requests

# Initialize the server
# uvicorn molrgen.server:app --reload

# Check liveness
response = requests.get("http://localhost:8000/liveness")
print(response.json())  # {"status": "ok"}

# Get molecular rewards
query_data = {
    "metadata": [
        {
            "properties": ["QED", "SA"],
            "objectives": ["maximize", "minimize"],
            "target": [0.0, 0.0]
        }
    ],
    "query": [
        "<answer>CC(C)Cc1ccc(cc1)C(C)C(=O)O</answer>",
    ]
}

response = requests.post("http://localhost:8000/get_reward", json=query_data)
result = response.json()
print(f"Overall reward: {result['reward']}")
print(f"Individual scores: {result['reward_list']}")
print(f"Feedback: {result['next_turn_feedback']}")
```

## Performance Considerations

- **Batching**: The server uses a buffering system (default 20 seconds) to batch multiple requests together for efficient GPU utilization. This can increase the time to perform a request. If you are working in a synchronous setting, consider decreasing the buffer time.
- **Concurrency**: Up to 2 docking runs per GPU by default.
