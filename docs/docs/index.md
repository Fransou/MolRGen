# MolGenDocking: Molecular Generation and Docking Benchmarks

Welcome to MolGenDocking, a comprehensive framework for molecular generation tasks with integrated protein-ligand docking evaluation. This project provides datasets, benchmarks, and a reward server for training and evaluating models that generate drug-like molecules optimized for specific biological targets.



## Quick Start

### Installation

**Basic Installation:**
```bash
git clone https://github.com/Fransou/MolGenDocking.git
cd MolGenDocking
pip install -e .[main]
```
*If called as `pip install .`, pytdc will not be installed.*

### Running the Reward Server

```bash
export DOCKING_ORACLE=autodock_gpu
... # Set other environment variables as needed
export DATA_PATH=... # Path to your data directory
uvicorn --host 0.0.0.0 --port 8000 mol_gen_docking.server:app
```

### Using the API

```python
import requests

response = requests.post(
    "http://localhost:8000/get_reward",
    json={
        "query": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "prompt": "Generate a drug-like molecule...",
        "metadata": [
            {
                "properties": ["QED", "protein_1"],
                "objectives": ["above", "minimize"],
                "target": [0.7, 0.0]
            }
        ]
    }
)
```

⚙️ **Reward Server API**
We use AutoDock-GPU for fast GPU-accelerated docking calculations.
The Molecular Verifier server is built using FastAPI, and supports concurrent requests, ensuring efficient handling of multiple docking evaluations, and asynchroneous pipelines.

## Citation

If you use MolGenDocking in your research, please cite:

```bibtex
...
```

## License

Apache License 2.0. See [LICENSE](https://github.com/Fransou/MolGenDocking/LICENSE.md) for details.

## Support

For issues, questions, or contributions, please visit our [GitHub repository](https://github.com/Fransou/MolGenDocking).
