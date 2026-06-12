"""
Experiments pipeline.

Auto-trains tokenizer + approach when checkpoints are missing, then runs all experiments.
Device is selected automatically (CUDA -> MPS -> CPU).

Experiments (run for every approach)
-------------------------------------
  I.  quality       — FID + Inception Score on the val split  → quality.csv
  II. speed_quality — FID + IS + samples/sec swept over temperatures → speed_quality.csv

Approaches
----------
  raster   — causal raster-order autoregressive transformer (unconditional)
  maskgit  — masked-token iterative generation (unconditional in debug config)
  var      — Visual AutoRegressive generation via multiscale image pyramid

=============================
 QUICK MODE  (--quick flag)
=============================
Limits FID/IS to 5 generated samples. Auto-trains if checkpoints are missing.

  -- Small models (smoke test, ~minutes) --

  # all 3 approaches at once
  uv run python scripts/run_pipeline.py --quick

  # raster only
  uv run python scripts/run_pipeline.py --quick --approaches configs/experiment/smoke_test_raster.yaml

  # maskgit only
  uv run python scripts/run_pipeline.py --quick --approaches configs/experiment/smoke_test_maskgit.yaml

  # var only
  uv run python scripts/run_pipeline.py --quick --approaches configs/experiment/smoke_test_var.yaml

  -- Full models, quick eval (pre-trained checkpoints required) --

  # all 3 approaches at once
  uv run python scripts/run_pipeline.py --quick \\
      --approaches configs/experiment/raster_pathmnist64_debug.yaml \\
                   configs/experiment/maskgit_pathmnist64_debug.yaml \\
                   configs/experiment/var_pathmnist64_d4.yaml

  # raster only
  uv run python scripts/run_pipeline.py --quick --approaches configs/experiment/raster_pathmnist64_debug.yaml

  # maskgit only
  uv run python scripts/run_pipeline.py --quick --approaches configs/experiment/maskgit_pathmnist64_debug.yaml

  # var only
  uv run python scripts/run_pipeline.py --quick --approaches configs/experiment/var_pathmnist64_d4.yaml

=============================
 FULL MODE  (no --quick)
=============================
FID/IS evaluated on the full val split (~10k images). Trains from scratch if needed.

  -- All 3 approaches at once --

  uv run python scripts/run_pipeline.py

  -- Single approach --

  # raster only
  uv run python scripts/run_pipeline.py --approaches configs/experiment/raster_pathmnist64_debug.yaml

  # maskgit only
  uv run python scripts/run_pipeline.py --approaches configs/experiment/maskgit_pathmnist64_debug.yaml

  # var only
  uv run python scripts/run_pipeline.py --approaches configs/experiment/var_pathmnist64_d4.yaml

=============================
 OTHER OPTIONS
=============================
  --split {train,val,test}       dataset split for loss evaluation (default: val)
  --quality-num-samples N        override FID/IS sample count
  --results-dir PATH             override output directory for CSV files
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ar_image_generation.config import load_experiment_config


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
    return parser.parse_args()


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd).returncode


def ensure_checkpoints(config: Path) -> bool:
    """Train tokenizer then approach if either checkpoint is missing. Returns True on success."""
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
    cmd = [
        sys.executable, "scripts/evaluate.py",
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


if __name__ == "__main__":
    main()
