"""Create individual-model or matched-intersection post-evaluation reports.

Examples:
  python -m robocasa.scripts.summarize_post_evaluation_metrics LEDGER --optimal OPTIMAL --out REPORT
  python -m robocasa.scripts.summarize_post_evaluation_metrics LEDGER1 LEDGER2 --matched-intersection --optimal OPTIMAL --out REPORT
"""
import argparse
import importlib.util
import json
from pathlib import Path

_SSI_PATH = Path(__file__).resolve().parents[1] / "metrics" / "ssi.py"
_SPEC = importlib.util.spec_from_file_location("robocasa_post_eval_ssi", _SSI_PATH)
_SSI = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SSI)
_DEFAULT_OPTIMAL = Path(__file__).resolve().parents[1] / "metrics" / "nonblocking_optimal_paths" / "path_length_time.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ledger_dirs", nargs="+")
    parser.add_argument("--matched-intersection", action="store_true")
    parser.add_argument("--optimal", default=str(_DEFAULT_OPTIMAL),
                        help="A* reference paths (default: bundled metrics artifact)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = (_SSI.summarize_blocking_intersection(args.ledger_dirs, args.optimal)
              if args.matched_intersection
              else _SSI.summarize_unpaired_ledger(args.ledger_dirs[0], args.optimal))
    with Path(args.out).open("w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
