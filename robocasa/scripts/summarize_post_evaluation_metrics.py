"""Write a post-evaluation metrics report from one or more ledgers."""
import argparse
import importlib.util
import json
from pathlib import Path


_CORE = Path(__file__).resolve().parents[1] / "metrics" / "summarize.py"
_SPEC = importlib.util.spec_from_file_location("robocasa_post_evaluation", _CORE)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger_dirs", nargs="+")
    parser.add_argument("--matched-intersection", action="store_true",
                        help="restrict all models to their common tasks")
    parser.add_argument("--optimal", default=_MOD.OPTIMAL)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = _MOD.summarize_post_evaluation(
        args.ledger_dirs, args.optimal,
        matched_intersection=args.matched_intersection)
    _MOD.check_post_evaluation(report)
    with Path(args.out).open("w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
