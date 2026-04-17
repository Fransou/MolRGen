import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.lines import Line2D

from notebooks.utils import (
    CMAP_MODELS,
    HIGHLIGHT_MODELS,
    MARKER_MODELS,
    get_top_k_div_df,
    load_molgen_results,
)


def load_config(config_path: Path | str) -> dict:
    """Load a YAML configuration file."""
    with open(config_path) as f:
        out: dict = yaml.safe_load(f)
    return out


def get_sim_values(sim_config: dict) -> list[float]:
    """Generate sim_values from config."""
    out: list[float]
    if sim_config["type"] == "logspace":
        out = np.logspace(
            sim_config["start"], sim_config["stop"], sim_config["num"]
        ).tolist()
    elif sim_config["type"] == "list":
        out = sim_config["values"]
    else:
        raise ValueError(f"Unknown sim_values type: {sim_config['type']}")
    return out


def get_x_vals(x_vals_config: dict | None) -> list[float] | None:
    """Generate x_vals from config."""
    if x_vals_config is None:
        return None
    result = []
    if "prepend" in x_vals_config:
        result.append(x_vals_config["prepend"])
    result.extend(
        [
            float(i)
            for i in range(
                x_vals_config["start"], x_vals_config["stop"], x_vals_config["step"]
            )
        ]
    )
    return result


def load_data(
    config: dict, save_file: str = "divtopk_data.csv"
) -> tuple[pd.DataFrame, list]:
    """Load and filter data based on config."""
    if os.path.exists(save_file):
        print(f"Loading preprocessed data from {save_file}...")
        df = pd.read_csv(save_file)
        df["smiles"] = df["smiles"].fillna("")
    else:
        molstral_path = Path(config["paths"]["molstral_path"])

        files = [
            f
            for d in molstral_path.iterdir()
            for f in d.iterdir()
            if "error" not in str(f) and str(f).endswith("_scored.jsonl")
        ]
        files = sorted(files)
        print("Total files:", len(files))
        df = load_molgen_results(files)
        df.to_csv(save_file, index=False)

    # Apply subsample filters
    sub_sample_prompts: list = []
    print(df.model.unique())
    if (
        config["data"]["subsample_models"] is not None
        and len(config["data"]["subsample_models"]) > 0
    ):
        for model in config["data"]["subsample_models"]:
            model_prompts = df[df.model == model].prompt_id.unique()
            df = df[df.prompt_id.isin(model_prompts)]
            sub_sample_prompts = model_prompts
    else:
        sub_sample_prompts = df.prompt_id.unique().tolist()
    df = df[~df.Model.isin(config["data"].get("remove_model", []))]
    return df, sub_sample_prompts[:]


def setup_paths(config: dict) -> tuple[str, str]:
    """Setup output paths based on config."""
    fig_path = config["paths"]["fig_path"]
    output_subdir = config["paths"]["output_subdir"]

    os.makedirs(fig_path, exist_ok=True)
    full_output_path = f"{fig_path}/{output_subdir}"
    os.makedirs(full_output_path, exist_ok=True)

    return fig_path, full_output_path


def plot_div_topk(
    df: pd.DataFrame,
    sub_sample_prompts: list,
    fp_name: str,
    legend: bool = False,
    cols_vals: list[float] = [10, 30],
    row_vals: list[float] = [1.5, 2, 4],
    x_vals: list[float] | None = None,
    row: str = "n_rollout_per_k",
    rollouts: list[int] = [32, 64, 128],
    rollouts_is_frac: bool = True,
    col: str = "k",
    x: str = "sim",
    column_name: str = "k",
    xlabel: str = "Similarity threshold between candidate clusters",
    xlabel_x: float = 0.3,
    ylabel_x: float = -0.01,
    legend_bbox: tuple[float, float] = (0.3, -0.2),
    legend_ncols: int = 4,
    height: float = 1.8,
    aspect: float = 2.5,
    markersize: dict[str, float] = {"normal": 5, "highlight": 7},
    **kwargs: Any,
) -> sns.FacetGrid:
    def draw(data: pd.DataFrame, **kwargs: Any) -> None:
        # Separate highlighted and non-highlighted models

        kwargs.update(
            dict(
                x=x,
                y="value",
                hue="Model",
                dashes=False,
                palette=CMAP_MODELS,
                hue_order=CMAP_MODELS.keys(),
            )
        )
        sns.lineplot(
            data,
            sizes=1,
            alpha=0.7,
            linewidth=0.5,
            legend=False,
            **kwargs,
        )
        sns.lineplot(
            data.iloc[2::3],
            style="Model",
            sizes=1,
            alpha=0.7,
            markers=MARKER_MODELS,
            markersize=markersize["normal"],
            linewidth=0.0,
            markeredgewidth=0.0,
            legend=legend,
            **kwargs,
        )

        # Draw highlighted models with enhanced styling (thicker lines, larger markers)
        if len(HIGHLIGHT_MODELS) > 0:
            for model in HIGHLIGHT_MODELS:
                highlighted_data = data[data["Model"] == model]
                sns.lineplot(
                    highlighted_data,
                    sizes=1,
                    alpha=1,
                    linewidth=1.6,
                    legend=False,
                    **kwargs,
                )
                sns.lineplot(
                    highlighted_data.iloc[2::3],
                    style="Model",
                    sizes=1,
                    alpha=1,
                    markers=MARKER_MODELS,
                    markersize=markersize["highlight"],
                    linewidth=0.0,
                    markeredgewidth=0.0,
                    legend=False,
                    **kwargs,
                )

    if row == "n_rollout_per_k":
        rollouts = np.unique([int(k * roll) for k in cols_vals for roll in row_vals])

    div_clus_df = get_top_k_div_df(
        df[df.prompt_id.isin(sub_sample_prompts[:])],
        fp_name=fp_name,
        rollouts=rollouts,
        **kwargs,
    )

    if row == "n_rollout_per_k":
        div_clus_df["n_rollout_per_k"] = div_clus_df["n_rollout"] / div_clus_df["k"]
        div_clus_df = div_clus_df[div_clus_df["n_rollout_per_k"].isin(row_vals)]

    if cols_vals is not None:
        div_clus_df = div_clus_df[div_clus_df[col].isin(cols_vals)]
    if x_vals is not None:
        div_clus_df = div_clus_df[div_clus_df[x].isin(x_vals)]
    g = sns.FacetGrid(
        div_clus_df,
        row=row,
        col=col,
        margin_titles=True,
        height=height,
        aspect=aspect,
        gridspec_kws={"wspace": 0.05, "hspace": 0.15},
    )
    g.map_dataframe(draw)
    # Add legend to the top right
    g.set_axis_labels("", "")
    # set x axis log
    g.set(xscale="log")
    if row == "n_rollout_per_k":
        row_temp = "$n_r/k$={row_name}"
    else:
        row_temp = "$n_r$={row_name}"
    g.set_titles(row_template=row_temp, col_template=column_name + "={col_name}")
    g.fig.supxlabel(xlabel, y=0.0, x=xlabel_x)
    g.fig.supylabel("Diversity-Aware Top-k Score", x=ylabel_x)

    if legend:
        handles = []
        labels = []
        for name, color in CMAP_MODELS.items():
            if name not in df.Model.unique():
                continue
            marker = MARKER_MODELS.get(name, None)
            is_highlight = name in HIGHLIGHT_MODELS
            lw = 1.4 if is_highlight else 1.0
            ms = markersize["highlight"] if is_highlight else markersize["normal"]

            if marker:
                handle = Line2D(
                    [0],
                    [0],
                    color=color,
                    lw=lw,
                    marker=marker,
                    markersize=ms,
                    markeredgewidth=0.0,
                )
            else:
                handle = Line2D([0], [0], color=color, lw=lw)

            handles.append(handle)
            labels.append(name)

        leg = g.fig.legend(
            handles=handles,
            labels=labels,
            title="Model",
            loc="lower center",
            bbox_to_anchor=legend_bbox,
            ncol=legend_ncols,
            fontsize=8,
            title_fontsize=10,
        )
        # Ensure legend background is fully opaque
        leg.get_frame().set_alpha(1.0)
    # Add grid
    for ax in g.axes.flatten():
        ax.grid(True, which="both", linestyle="--", linewidth=0.05, alpha=0.5)
    return g


def run_figure(config_path: Path, display: bool) -> None:
    """Generate figure(s) from config. Handles single or multiple fingerprints."""
    config = load_config(config_path)
    df, sub_sample_prompts = load_data(config)
    fig_path, full_output_path = setup_paths(config)

    plot_config = config["plot"]
    sim_values = get_sim_values(config["sim_values"])
    output_suffix = config["paths"]["output_suffix"]

    # Determine fingerprints to iterate over
    fingerprints = plot_config.get("fingerprints", [plot_config.get("fp_name")])
    legend_fingerprint = plot_config.get("legend_fingerprint")

    # Get x_vals from range config if present, otherwise from direct config
    x_vals = get_x_vals(plot_config.get("x_vals_range")) or plot_config.get("x_vals")

    for fp_name in fingerprints:
        # Determine legend settings for this fingerprint
        if legend_fingerprint is not None:
            has_legend = fp_name == legend_fingerprint
        else:
            has_legend = plot_config.get("legend", False)

        # Determine xlabel_x based on legend
        if has_legend and "xlabel_x_with_legend" in plot_config:
            xlabel_x = plot_config["xlabel_x_with_legend"]
            ylabel_x = plot_config["ylabel_x_with_legend"]
        elif "xlabel_x_without_legend" in plot_config:
            xlabel_x = plot_config["xlabel_x_without_legend"]
            ylabel_x = plot_config["ylabel_x_without_legend"]
        else:
            xlabel_x = plot_config.get("xlabel_x", 0.5)
            ylabel_x = plot_config.get("ylabel_x", -0.01)

        # Build kwargs for plot_div_topk
        plot_kwargs = {
            "df": df,
            "sub_sample_prompts": sub_sample_prompts,
            "fp_name": fp_name,
            "cols_vals": plot_config["cols_vals"],
            "x_vals": x_vals,
            "row": plot_config["row"],
            "rollouts": plot_config.get("rollouts", [32, 64, 128]),
            "col": plot_config["col"],
            "x": plot_config["x"],
            "column_name": plot_config["column_name"],
            "xlabel": plot_config["xlabel"],
            "xlabel_x": xlabel_x,
            "ylabel_x": ylabel_x,
            "legend": has_legend,
            "legend_bbox": tuple(plot_config["legend_bbox"]),
            "legend_ncols": plot_config["legend_ncols"],
            "height": plot_config["height"],
            "aspect": plot_config["aspect"],
            "markersize": plot_config["markersize"],
            "sim_values": sim_values,
        }

        # Add optional k_max if present
        if "k_max" in plot_config:
            plot_kwargs["k_max"] = plot_config["k_max"]

        g = plot_div_topk(**plot_kwargs)

        # Add suptitle if configured (for secondary figures)
        if "suptitle_y" in plot_config:
            g.fig.suptitle(fp_name, y=plot_config["suptitle_y"])

        output_file = f"{full_output_path}/{fp_name}{output_suffix}.pdf"
        plt.savefig(output_file, bbox_inches="tight")
        if display:
            plt.show()
        else:
            plt.clf()
            plt.close()
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    import argparse

    CONFIG_DIR = Path(__file__).parent / "configs"

    parser = argparse.ArgumentParser(
        description="Generate DivTopK plots from YAML configs"
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["main", "secondary", "last", "all"],
        default="all",
        help="Which config to run (default: all)",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to a custom config file (overrides --config)",
    )

    parser.add_argument(
        "--display", action="store_true", help="Display plots interactively"
    )

    args = parser.parse_args()

    if args.config_file:
        run_figure(args.config_file, args.display)
    else:
        if args.config in ["main", "all"]:
            print("=" * 50)
            print("Running main figure...")
            print("=" * 50)
            run_figure(CONFIG_DIR / "main_figure.yaml", args.display)

        if args.config in ["secondary", "all"]:
            print("=" * 50)
            print("Running secondary figures...")
            print("=" * 50)
            run_figure(CONFIG_DIR / "secondary_figures.yaml", args.display)

        if args.config in ["last", "all"]:
            print("=" * 50)
            print("Running last figure...")
            print("=" * 50)
            run_figure(CONFIG_DIR / "last_figure.yaml", args.display)
