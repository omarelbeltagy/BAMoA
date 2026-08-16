# BA-MoA unified experiment runner.
#
# Single entry point for both dataset runners. Default behavior: auto-detect
# the most recent output file for the chosen dataset and continue from it;
# use --fresh to force starting a brand-new run instead.
import argparse
import asyncio
import glob
import os

import bbq_runner
import winobias_runner


def find_latest_run(output_dir):
    """Return the path to the most recent run_*.json in output_dir, or None
    if the directory doesn't exist or has no run files. Filenames are
    timestamp-formatted (run_YYYYMMDD_HHMMSS.json), so lexical sort order
    matches chronological order."""
    if not os.path.isdir(output_dir):
        return None
    candidates = sorted(glob.glob(os.path.join(output_dir, "run_*.json")))
    return candidates[-1] if candidates else None


def build_runner_argv(args, latest_path):
    """Translate app-level args into the argv list a runner's main(argv)
    expects, mirroring each runner's own --continue / --n-per-cell / etc."""
    argv = []
    if latest_path and not args.fresh:
        argv += ["--continue", latest_path]
    if args.n_per_cell is not None:
        argv += ["--n-per-cell", str(args.n_per_cell)]
    if args.dataset == "winobias" and args.include_type_2:
        argv += ["--include-type-2"]
    return argv


async def run_dataset(args):
    output_dir = os.path.join("outputs", args.dataset)
    latest_path = None if args.fresh else find_latest_run(output_dir)

    if latest_path:
        print(f"[app] Found existing run for '{args.dataset}': {latest_path}")
        print(f"[app] Continuing from it (use --fresh to start a new run instead).")
    else:
        print(f"[app] No existing run found for '{args.dataset}' — starting fresh.")

    argv = build_runner_argv(args, latest_path)
    print(f"[app] Invoking {args.dataset} runner with argv={argv}")

    if args.dataset == "bbq":
        await bbq_runner.main(argv)
    elif args.dataset == "winobias":
        await winobias_runner.main(argv)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def parse_app_args():
    parser = argparse.ArgumentParser(description="BA-MoA unified experiment runner")
    parser.add_argument(
        "--dataset", choices=["bbq", "winobias"], required=True,
        help="Which dataset/experiment to run.",
    )
    parser.add_argument(
        "--n-per-cell", type=int, default=None,
        help="Override the dataset runner's default n-per-cell "
             "(omit to use each runner's own default).",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Start a brand-new run even if a previous run file exists for "
             "this dataset. Default behavior auto-continues from the latest "
             "existing run.",
    )
    parser.add_argument(
        "--include-type-2", action="store_true",
        help="(winobias only) also run the fixed Type-2 sanity check.",
    )
    return parser.parse_args()


def main():
    args = parse_app_args()
    asyncio.run(run_dataset(args))


if __name__ == "__main__":
    main()