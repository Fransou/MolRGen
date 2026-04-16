import seaborn as sns

from .molgen_utils import (
    aggregate_molgen_fn,
    fp_name_to_fn,
    get_top_k_div_df,
    load_molgen_results,
)
from .molprop_utils import load_molprop_results
from .pandas_to_latex import PandasTableFormatter
from .synthesis_utils import load_synth_results

N_COLORS = 14
palette = sns.husl_palette(
    n_colors=N_COLORS,
    s=0.95,  # Saturation
    l=0.62,  # Lightness
)
CMAP_MODELS = {
    # Chem models — muted reds
    "ChemDFM-R": palette[0],
    "ChemDFM-v2.0": palette[1],
    "RL-Mistral": palette[-1],
    "RL-Mistral-100": palette[-2],
    "RL-Molstral": palette[-3],
    "ether0": palette[2],
}

CMAP_MODELS.update(
    {
        # Reasoning models — cool blues / teals
        "MiniMax-M2": palette[4],
        "Qwen3": palette[5],
        "Qwen3-Next": palette[6],
        "gpt-oss": palette[7],
        "R1-Llama": palette[8],
        "R1-Qwen": palette[9],
    }
)

CMAP_MODELS.update(
    {
        # Non-reasoning models — warm greens / golds
        "Llama-3.3": palette[10],
        "gemma-3": palette[11],
    }
)
MARKER_MODELS = {
    # Chemistry-specialized models
    "RL-Mistral": "X",
    "RL-Mistral-100": "X",
    "RL-Molstral": "X",
    "ChemDFM-R": "^",
    "ChemDFM-v2.0": "o",
    "ether0": "v",
    # Reasoning models
    "MiniMax-M2": "h",
    "Qwen3": "s",
    "Qwen3-Next": "^",
    "gpt-oss": "D",
    "R1-Llama": "P",
    "R1-Qwen": "X",
    # Non-reasoning general models
    "Llama-3.3": "p",
    "gemma-3": "*",
}

# Models to highlight in plots
HIGHLIGHT_MODELS = ["ChemDFM-R", "MiniMax-M2", "RL-Mistral", "RL-Molstral"]

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "PandasTableFormatter",
    "load_molgen_results",
    "aggregate_molgen_fn",
    "get_top_k_div_df",
    "CMAP_MODELS",
    "MARKER_MODELS",
    "HIGHLIGHT_MODELS",
    "fp_name_to_fn",
    "load_molprop_results",
    "load_synth_results",
]
