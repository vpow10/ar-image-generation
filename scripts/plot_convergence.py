"""
Experiment 4 — Learning curves (convergence).

Reads training metrics saved by train_approach.py and produces a comparison
plot of validation loss vs epoch for all approaches on the same axes.
Also saves a tidy CSV with all curves for further analysis.

Usage (standalone):
  uv run python scripts/plot_convergence.py \\
      --configs configs/experiment/raster_pathmnist64_debug.yaml \\
               configs/experiment/maskgit_pathmnist64_debug.yaml \\
               configs/experiment/var_pathmnist64_d4.yaml \\
      --output-dir runs/results

  # quick mode (reads smoke-test run dirs):
  uv run python scripts/plot_convergence.py \\
      --configs configs/experiment/smoke_test_raster.yaml \\
               configs/experiment/smoke_test_maskgit.yaml \\
               configs/experiment/smoke_test_var.yaml \\
      --output-dir runs/results/quick
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ar_image_generation.config import load_experiment_config
from ar_image_generation.evaluation.io import append_csv_row


APPROACH_COLORS = {
    "raster": "#4C72B0",
    "maskgit": "#DD8452",
    "var": "#55A868",
}
FALLBACK_COLORS = ["#C44E52", "#8172B2", "#937860", "#DA8BC3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot learning curves (Experiment 4).")
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        required=True,
        help="Experiment config YAMLs (one per approach).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/results"),
        help="Directory for output PNG and CSV (default: runs/results).",
    )
    parser.add_argument(
        "--no-train-loss",
        action="store_true",
        help="Omit train loss curves — show only validation loss.",
    )
    return parser.parse_args()


def load_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return sorted(records, key=lambda r: r["epoch"])


def plot_convergence(
    *,
    configs: list[Path],
    output_dir: Path,
    show_train_loss: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    fallback_idx = 0
    csv_path = output_dir / "convergence.csv"

    # Remove stale CSV so header is re-written cleanly.
    if csv_path.exists():
        csv_path.unlink()

    any_data = False

    for config_path in configs:
        cfg = load_experiment_config(config_path)
        run_name = cfg.logging.run_name
        approach = cfg.approach.name
        run_dir = Path("runs") / run_name
        records = load_metrics(run_dir)

        if not records:
            print(f"  [E4] No metrics found for '{run_name}' — skipping.")
            continue

        any_data = True
        epochs = [r["epoch"] for r in records]
        val_losses = [r["val"]["loss"] for r in records]
        train_losses = [r["train"]["loss"] for r in records]

        color = APPROACH_COLORS.get(approach)
        if color is None:
            color = FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
            fallback_idx += 1

        ax.plot(epochs, val_losses, color=color, linewidth=2, label=f"{approach} (val)")
        if show_train_loss:
            ax.plot(
                epochs, train_losses,
                color=color, linewidth=1, linestyle="--",
                alpha=0.5, label=f"{approach} (train)",
            )

        for epoch, train_l, val_l in zip(epochs, train_losses, val_losses):
            append_csv_row(
                {
                    "run_name": run_name,
                    "approach": approach,
                    "epoch": epoch,
                    "train_loss": round(train_l, 6),
                    "val_loss": round(val_l, 6),
                },
                csv_path,
            )

    if not any_data:
        print("  [E4] No training data found — train models first.")
        plt.close(fig)
        return output_dir

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation loss", fontsize=12)
    ax.set_title("E4 — Learning curves: validation loss vs epoch", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()

    png_path = output_dir / "convergence.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"  [E4] Plot saved → {png_path}")
    print(f"  [E4] CSV  saved → {csv_path}")
    return png_path


def main() -> None:
    args = parse_args()
    plot_convergence(
        configs=args.configs,
        output_dir=args.output_dir,
        show_train_loss=not args.no_train_loss,
    )


if __name__ == "__main__":
    main()
