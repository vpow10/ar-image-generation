"""
Experiments pipeline.

Auto-trains tokenizer + approach when checkpoints are missing, then runs all experiments.
Device is selected automatically (CUDA → MPS → CPU).

Experiments
-----------
  I.   quality          — FID + Inception Score on the val split       → quality.csv
  II.  speed_quality    — FID + IS swept over temperatures + speed     → speed_quality.csv
  III. tokenizer_quality— PSNR + SSIM on encode→decode reconstruction  → tokenizer_quality.csv
  IV.  convergence      — val loss vs epoch for all approaches combined → convergence.png / convergence.csv
                          (requires metrics.jsonl written during training)

Approaches (standard — use evaluate.py)
-----------------------------------------
  raster   — causal raster-order autoregressive transformer
  maskgit  — masked-token iterative generation
  var      — Visual AutoRegressive generation via multiscale image pyramid

Custom approach (uses evaluate_custom.py automatically)
--------------------------------------------------------
  custom   — GRAFT-GS Gaussian splatting prior (GaussianPrimitiveAR + GaussianSplatTokenizer)
             Checkpoint expected at: runs/<run_name>/checkpoints/best.pt
             Training: scripts/train_graft_gs_prior.py  (separate, not auto-triggered)

Results layout
--------------
  runs/results[/quick]/<config_stem>/quality.csv
  runs/results[/quick]/<config_stem>/speed_quality.csv
  runs/results[/quick]/<config_stem>/tokenizer_quality.csv
  runs/results[/quick]/convergence.png
  runs/results[/quick]/convergence.csv

=============================
 QUICK MODE  (--quick flag)
=============================
Limits FID/IS to 5 generated samples. Auto-trains standard models if checkpoints missing.

  -- Small models (smoke test, ~minutes) --

  # all 3 standard approaches + Exp 4
  uv run python scripts/run_pipeline.py --quick

  # single approach, Exp 1–3 only
  uv run python scripts/run_pipeline.py --quick --no-convergence \\
      --approaches configs/experiment/smoke_test_raster.yaml

  uv run python scripts/run_pipeline.py --quick --no-convergence \\
      --approaches configs/experiment/smoke_test_maskgit.yaml

  uv run python scripts/run_pipeline.py --quick --no-convergence \\
      --approaches configs/experiment/smoke_test_var.yaml

  # Exp 4 only (learning curves), smoke test models
  uv run python scripts/run_pipeline.py --quick --convergence-only

  -- Full models, quick eval (pre-trained checkpoints required) --

  # all 3 standard approaches + Exp 4
  uv run python scripts/run_pipeline.py --quick \\
      --approaches configs/experiment/raster_pathmnist64_debug.yaml \\
                   configs/experiment/maskgit_pathmnist64_debug.yaml \\
                   configs/experiment/var_pathmnist64_d4.yaml

  # all 4 approaches (including custom) + Exp 4
  uv run python scripts/run_pipeline.py --quick \\
      --approaches configs/experiment/raster_pathmnist64_debug.yaml \\
                   configs/experiment/maskgit_pathmnist64_debug.yaml \\
                   configs/experiment/var_pathmnist64_d4.yaml \\
                   configs/experiment/custom_graft_gs_prior_pathmnist64.yaml

  # single approach, Exp 1–3 only
  uv run python scripts/run_pipeline.py --quick --no-convergence \\
      --approaches configs/experiment/raster_pathmnist64_debug.yaml

=============================
 FULL MODE  (no --quick)
=============================
FID/IS evaluated on the full val split (~10k images). Trains standard models from scratch if needed.

  # all 3 standard approaches + Exp 4
  uv run python scripts/run_pipeline.py \\
      --approaches configs/experiment/raster_pathmnist64_debug.yaml \\
                   configs/experiment/maskgit_pathmnist64_debug.yaml \\
                   configs/experiment/var_pathmnist64_d4.yaml

  # all 4 approaches (including custom) + Exp 4
  uv run python scripts/run_pipeline.py \\
      --approaches configs/experiment/raster_pathmnist64_debug.yaml \\
                   configs/experiment/maskgit_pathmnist64_debug.yaml \\
                   configs/experiment/var_pathmnist64_d4.yaml \\
                   configs/experiment/custom_graft_gs_prior_pathmnist64.yaml

  # single approach, Exp 1–3 only
  uv run python scripts/run_pipeline.py --no-convergence \\
      --approaches configs/experiment/raster_pathmnist64_debug.yaml

  uv run python scripts/run_pipeline.py --no-convergence \\
      --approaches configs/experiment/maskgit_pathmnist64_debug.yaml

  uv run python scripts/run_pipeline.py --no-convergence \\
      --approaches configs/experiment/var_pathmnist64_d4.yaml

  # Exp 4 only (learning curves), full models
  uv run python scripts/run_pipeline.py --convergence-only \\
      --approaches configs/experiment/raster_pathmnist64_debug.yaml \\
                   configs/experiment/maskgit_pathmnist64_debug.yaml \\
                   configs/experiment/var_pathmnist64_d4.yaml

=============================
 OTHER OPTIONS
=============================
  --no-convergence               skip Exp 4 (useful when running a single approach)
  --convergence-only             run only Exp 4; skip training checks and Exp 1–3
  --split {train,val,test}       dataset split for loss evaluation (default: val)
  --quality-num-samples N        override FID/IS sample count
  --results-dir PATH             override output directory for CSV files
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from ar_image_generation.config import load_experiment_config, load_yaml


QUICK_CONFIGS = [
    Path("configs/experiment/smoke_test_raster.yaml"),
    Path("configs/experiment/smoke_test_maskgit.yaml"),
    Path("configs/experiment/smoke_test_var.yaml"),
]
QUICK_QUALITY_SAMPLES = 5

FULL_APPROACHES = [
    Path("configs/experiment/raster_pathmnist64_debug.yaml"),
    Path("configs/experiment/maskgit_pathmnist64_debug.yaml"),
    Path("configs/experiment/var_pathmnist64_d4.yaml"),
    Path("configs/experiment/custom_graft_gs_prior_pathmnist64.yaml"),
]


@dataclass
class Experiment:
    name: str
    csv_name: str
    extra_args: list[str]


EXPERIMENTS: list[Experiment] = [
    Experiment(
        name="quality",
        csv_name="quality.csv",
        extra_args=["--quality", "--quality-split", "val", "--quality-batch-size", "64"],
    ),
    Experiment(
        name="speed_quality",
        csv_name="speed_quality.csv",
        extra_args=["--speed-quality", "--quality-split", "val", "--quality-batch-size", "64"],
    ),
    Experiment(
        name="tokenizer_quality",
        csv_name="tokenizer_quality.csv",
        extra_args=["--tokenizer-quality", "--tokenizer-quality-split", "test"],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the experiment pipeline.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: smoke_test_{raster,maskgit,var} configs, 5 quality samples. "
             "Auto-trains if checkpoints are missing.",
    )
    parser.add_argument(
        "--approaches",
        nargs="+",
        type=Path,
        default=None,
        help="Override approach configs. Defaults: smoke_test (--quick) or full list.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for per-experiment CSV files. "
             "Defaults: runs/results/quick or runs/results.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split for loss evaluation.",
    )
    parser.add_argument(
        "--quality-num-samples",
        type=int,
        default=None,
        help="Override number of images for FID/IS. "
             "Defaults: 5 (--quick) or full val split.",
    )
    parser.add_argument(
        "--no-convergence",
        action="store_true",
        help="Skip Experiment 4 (learning curves plot).",
    )
    parser.add_argument(
        "--convergence-only",
        action="store_true",
        help="Run only Experiment 4 (learning curves plot). "
             "Skips training checks and Experiments 1–3.",
    )
    return parser.parse_args()


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd).returncode


def is_custom_config(config: Path) -> bool:
    """Return True if the config cannot be parsed as a standard ExperimentConfig."""
    try:
        load_experiment_config(config)
        return False
    except (ValidationError, Exception):
        return True


def ensure_checkpoints(config: Path) -> bool:
    """Train tokenizer then approach if either checkpoint is missing. Returns True on success."""
    if is_custom_config(config):
        raw = load_yaml(config)
        run_name = raw["logging"]["run_name"]
        approach_ckpt = Path("runs") / run_name / "checkpoints" / "best.pt"
        if not approach_ckpt.exists():
            print(f"  Custom approach checkpoint missing: {approach_ckpt}")
            print(f"  Train manually with the appropriate graft-gs script, then re-run.")
            return False
        return True

    cfg = load_experiment_config(config)

    if not cfg.tokenizer.checkpoint_path.exists():
        print(f"  Tokenizer checkpoint missing — training ({config.name}) ...")
        rc = _run([sys.executable, "scripts/train_tokenizer.py", "--config", str(config)])
        if rc != 0:
            return False

    approach_ckpt = Path("checkpoints") / "approaches" / cfg.approach.name / "best.pt"
    if not approach_ckpt.exists():
        print(f"  Approach checkpoint missing — training ({config.name}) ...")
        rc = _run([sys.executable, "scripts/train_approach.py", "--config", str(config)])
        if rc != 0:
            return False

    return True


def run_experiment(
    *,
    approach_config: Path,
    experiment: Experiment,
    results_dir: Path,
    split: str,
    quality_num_samples: int | None,
) -> int:
    approach_name = approach_config.stem
    csv_path = results_dir / approach_name / experiment.csv_name
    evaluate_script = (
        "scripts/evaluate_custom.py"
        if is_custom_config(approach_config)
        else "scripts/evaluate.py"
    )
    cmd = [
        sys.executable, evaluate_script,
        "--config", str(approach_config),
        "--split", split,
        "--csv", str(csv_path),
        *experiment.extra_args,
    ]
    if quality_num_samples is not None:
        cmd += ["--quality-num-samples", str(quality_num_samples)]
    return _run(cmd)


def main() -> None:
    args = parse_args()

    # Resolve mode-dependent defaults
    approaches = args.approaches or (QUICK_CONFIGS if args.quick else FULL_APPROACHES)
    results_dir = args.results_dir or Path("runs/results/quick" if args.quick else "runs/results")
    quality_num_samples = args.quality_num_samples or (QUICK_QUALITY_SAMPLES if args.quick else None)

    results_dir.mkdir(parents=True, exist_ok=True)

    # --convergence-only: skip Exp 1–3, go straight to the plot.
    if args.convergence_only:
        print(f"\n{'=' * 60}")
        print("Experiment 4: Learning curves  (convergence-only mode)")
        print(f"{'=' * 60}")
        _run([
            sys.executable, "scripts/plot_convergence.py",
            "--configs", *[str(c) for c in approaches],
            "--output-dir", str(results_dir),
        ])
        return

    total = len(approaches) * len(EXPERIMENTS)
    done = 0
    failed: list[str] = []

    for approach_config in approaches:
        approach_name = approach_config.stem
        print(f"\n{'=' * 60}")
        print(f"Approach: {approach_name}  ({'quick' if args.quick else 'normal'})")
        print(f"{'=' * 60}")

        if not ensure_checkpoints(approach_config):
            print(f"  Could not prepare checkpoints for {approach_name} — skipping.")
            failed.append(f"{approach_name}/(training)")
            continue

        for experiment in EXPERIMENTS:
            done += 1
            print(f"\n[{done}/{total}] experiment={experiment.name}  approach={approach_name}")

            rc = run_experiment(
                approach_config=approach_config,
                experiment=experiment,
                results_dir=results_dir,
                split=args.split,
                quality_num_samples=quality_num_samples,
            )

            if rc != 0:
                failed.append(f"{approach_name}/{experiment.name}")
                print(f"  FAILED (exit {rc}) — skipping remaining experiments for this approach.")
                break

    # Experiment 4 — learning curves (runs once for all approaches together).
    if not args.no_convergence:
        print(f"\n{'=' * 60}")
        print("Experiment 4: Learning curves")
        print(f"{'=' * 60}")
        _run([
            sys.executable, "scripts/plot_convergence.py",
            "--configs", *[str(c) for c in approaches],
            "--output-dir", str(results_dir),
        ])

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete  ({done}/{total} runs,  {len(failed)} failed)")
    if failed:
        print("Failed:")
        for tag in failed:
            print(f"  {tag}")
    else:
        print("Results:")
        for approach_config in approaches:
            approach_name = approach_config.stem
            for experiment in EXPERIMENTS:
                csv_path = results_dir / approach_name / experiment.csv_name
                if csv_path.exists():
                    print(f"  {approach_name}/{experiment.csv_name}: {csv_path}")
        convergence_png = results_dir / "convergence.png"
        if convergence_png.exists():
            print(f"  convergence plot: {convergence_png}")


if __name__ == "__main__":
    main()
