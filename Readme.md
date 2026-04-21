[![Tests-PIP](https://github.com/Fransou/MolRGen/actions/workflows/test_pip_installation.yml/badge.svg)](https://github.com/Fransou/MolRGen/actions/workflows/test_pip_installation.yml)
[![Tests-DOCKER](https://github.com/Fransou/MolRGen/actions/workflows/test_docker_image.yml/badge.svg)](https://github.com/Fransou/MolRGen/actions/workflows/test_docker_image.yml)
[![Documentation](https://github.com/Fransou/MolRGen/actions/workflows/docs.yml/badge.svg)](https://fransou.github.io/MolRGen/)

# MolRGen: Molecular Generation and Docking Benchmarks

A comprehensive framework for molecular generation tasks with integrated protein-ligand docking evaluation. This project provides datasets, benchmarks, and a reward server for training and evaluating models that generate drug-like molecules optimized for specific biological targets.

📚 **[Full Documentation](https://fransou.github.io/MolRGen/)**
📚 **[HuggingFace Dataset](https://huggingface.co/datasets/Franso/MolRGen)

## :nut_and_bolt: Installation

```bash
git clone https://github.com/Fransou/MolRGen.git
cd MolRGen
pip install -e .[main]
```


**Docker** (recommended):
```bash
docker pull fransou/molgendata:latest
```

## :rocket: Quick Start

### Running the Reward Server

```bash
export DOCKING_ORACLE=autodock_gpu
export DATA_PATH=data
uvicorn --host 0.0.0.0 --port 8000 molrgen.server:app
```

### Example API Call

```python
import requests

response = requests.post(
    "http://localhost:8000/get_reward",
    json={
        "query": "<answer>CC(C)Cc1ccc(cc1)C(C)C(=O)O</answer>",
        "prompt": "Generate a drug-like molecule...",
        "metadata": [{
            "properties": ["QED", "docking_target"],
            "objectives": ["above", "minimize"],
            "target": [0.7, 0.0]
        }]
    }
)
```

## :books: Documentation

| Section | Description |
|---------|-------------|
| [Datasets](https://fransou.github.io/MolRGen/datasets/main/) | De novo generation, property prediction, and retro-synthesis datasets |
| [Reward Server](https://fransou.github.io/MolRGen/reward_server/getting_started/) | Server configuration, endpoints, and query format |
| [API Reference](https://fransou.github.io/MolRGen/api/toc/) | Detailed API documentation for all modules |

## :microscope: Supported Tasks

- **De Novo Molecular Generation**: Generate molecules optimizing docking scores, drug-likeness (QED, SA), and physicochemical properties
- **Property Prediction**: Regression and classification tasks using Polaris datasets
- **Retro-Synthesis**: Reaction prediction using USPTO data

## Citation

```bibtex
Ongoing work
```

## License

[Apache 2.0 License](LICENSE)
