"""Roll the ledgers under one outputs folder up into the headline numbers.

    python3 -m robocasa.metrics.summarize <folder>

`<folder>` is a run directory — an absolute path, or a name under
`ROBOCASA_OUTPUTS`. Every ledger beneath it is found and read; a sweep that
wrote one ledger per task and a single-episode check both work, and nothing
has to be named in advance.

This reads the ledgers that `robocasa.logging.log` writes DURING the run,
not the saved logs. `parse_run_logs.py` covers the other direction — parsing run.log
after the fact — and the two are kept separate because they fail differently:
extraction is a regex against text that changed shape over time, this is a
jsonl the runner wrote itself.

Three numbers come out:

    task_success_rate            reached the goal pose
    collision_free_success_rate  reached it without touching the obstacle
    normalized_path              the A* optimal length for this (layout,
                                 route) over the length actually driven

The optimal lengths come from the non-blocking sweep, which planned one path
per (layout, route) — the obstacle does not enter the reference, so a Blocking
episode is measured against the unobstructed plan and pays for every metre it
spends going around.

normalized_path exceeds 1 whenever the robot stopped short: the denominator is
what it drove, not what it had to drive. Read it next to
task_success_rate — on its own a run that gives up early scores well.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Same default as parse_run_logs.py, and the same env var. Two modules in one package
# disagreeing about where runs live is how a report ends up empty.
OUT_ROOT = os.environ.get("ROBOCASA_OUTPUTS", os.path.join(
    os.getcwd(), "policy", "Voxposer", "outputs"))

# The A* reference, written by the non-blocking optimal-path sweep.
OPTIMAL = os.environ.get("ROBOCASA_OPTIMAL_PATHS", os.path.join(
    os.getcwd(), "outputs", "nonblocking_optimal_paths", "path_length_time.json"))


def find_ledgers(root):
    """Every directory under `root` that holds an episodes.jsonl, `root` included."""
    root = Path(root)
    hits = sorted({p.parent for p in root.rglob("episodes.jsonl")})
    return hits


def load_planned(path):
    """(layout_name, route) -> A* path length in metres."""
    cells = json.load(open(path))["cells"]
    return {(c["layout_name"], c["route"]): c["planned_path_len_m"] for c in cells}


def route_of(task):
    """`NavigateKitchenDogBlockingRouteA` -> `RouteA`, matching the optimal json."""
    return "Route" + task.rsplit("Route", 1)[1] if "Route" in (task or "") else None


def executed_length(ledger, ep):
    """(metres, source) driven by this episode.

    The episode's own evaluation.json is preferred: it sums every control step.
    The ledger's trajectory holds one sample in `log_interval` steps, so the
    length recomputed from it runs a few percent short — usable, but the source
    is reported so two runs are never compared across the two.
    """
    run = ledger / (ep.get("run_dir") or "") / "evaluation.json"
    if ep.get("run_dir") and run.exists():
        value = json.load(open(run)).get("path_length_m")
        if value is not None:
            return float(value), "control-step"
    traj = ledger / "traj" / f"{ep['id']}.npz"
    if traj.exists():
        pos = np.load(traj)["pos_xy"]
        return float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))), "sampled"
    return None, None


def read_ledger(ledger, planned):
    """One record per episode, with the normalized path attached where it exists."""
    rows = []
    for line in open(ledger / "episodes.jsonl"):
        if not line.strip():
            continue
        ep = json.loads(line)
        ref = planned.get((ep.get("layout"), route_of(ep.get("task"))))
        actual, source = executed_length(ledger, ep)
        rows.append({
            **ep,
            "ledger": str(ledger),
            "planned_path_len_m": ref,
            "executed_path_len_m": actual,
            "path_length_source": source,
            "normalized_path": (ref / actual) if ref and actual else None,
        })
    return rows


def aggregate(rows):
    """The three rates over a list of episode records."""
    def rate(key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return (sum(bool(x) for x in v) / len(v), len(v)) if v else (None, 0)

    npath = [r["normalized_path"] for r in rows if r["normalized_path"] is not None]
    tsr, n_tsr = rate("task_success")
    csr, n_csr = rate("collision_free_success")
    return {
        "episodes": len(rows),
        "task_success_rate": tsr, "task_success_decided": n_tsr,
        "collision_free_success_rate": csr, "collision_free_success_decided": n_csr,
        "normalized_path": float(np.mean(npath)) if npath else None,
        "normalized_path_n": len(npath),
        # A missing reference is not a zero. It means the optimal sweep never
        # planned this (layout, route), and saying so beats a silent gap.
        "no_reference": sum(1 for r in rows if r["planned_path_len_m"] is None),
    }


def fmt(x, width=7, pct=False):
    if x is None:
        return f"{'n/a':>{width}}"
    return f"{x:{width}.1%}" if pct else f"{x:{width}.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder",
                    help="run directory: an absolute path, or a name under "
                         "ROBOCASA_OUTPUTS")
    ap.add_argument("--optimal", default=OPTIMAL,
                    help="optimal-path json (env: ROBOCASA_OPTIMAL_PATHS)")
    ap.add_argument("--episodes", action="store_true",
                    help="also print one line per episode")
    ap.add_argument("--json", default="",
                    help="also write the full result to this path")
    a = ap.parse_args()

    root = Path(a.folder)
    if not root.is_absolute() and not root.exists():
        root = Path(OUT_ROOT) / a.folder
    if not root.exists():
        sys.exit(f"no such folder: {root}")
    if not os.path.exists(a.optimal):
        sys.exit(f"optimal-path json not found: {a.optimal}\n"
                 f"point --optimal or ROBOCASA_OPTIMAL_PATHS at it")

    planned = load_planned(a.optimal)
    ledgers = find_ledgers(root)
    if not ledgers:
        sys.exit(f"no episodes.jsonl under {root}")

    per_ledger, rows = {}, []
    for ledger in ledgers:
        got = read_ledger(ledger, planned)
        if not got:
            continue
        per_ledger[str(ledger)] = aggregate(got)
        rows += got

    print(f"folder : {root}")
    print(f"optimal: {a.optimal}  ({len(planned)} cells)")
    print(f"ledgers: {len(per_ledger)}  episodes: {len(rows)}")
    print()
    hdr = (f"{'ledger':<58}{'n':>5}{'task_succ':>11}{'coll_free':>11}"
           f"{'norm_path':>11}")
    print(hdr)
    print("-" * len(hdr))
    for name, s in per_ledger.items():
        short = name[-57:] if len(name) > 57 else name
        print(f"{short:<58}{s['episodes']:>5}"
              f"{fmt(s['task_success_rate'], 11, pct=True)}"
              f"{fmt(s['collision_free_success_rate'], 11, pct=True)}"
              f"{fmt(s['normalized_path'], 11)}")
    total = aggregate(rows)
    print("-" * len(hdr))
    print(f"{'TOTAL':<58}{total['episodes']:>5}"
          f"{fmt(total['task_success_rate'], 11, pct=True)}"
          f"{fmt(total['collision_free_success_rate'], 11, pct=True)}"
          f"{fmt(total['normalized_path'], 11)}")

    print()
    print(f"task_success_rate           {fmt(total['task_success_rate'], 8, pct=True)}"
          f"   (decided {total['task_success_decided']}/{total['episodes']})")
    print(f"collision_free_success_rate {fmt(total['collision_free_success_rate'], 8, pct=True)}"
          f"   (decided {total['collision_free_success_decided']}/{total['episodes']})")
    print(f"normalized_path             {fmt(total['normalized_path'], 8)}"
          f"   (planned/actual over {total['normalized_path_n']}/{total['episodes']})")
    if total["no_reference"]:
        print(f"  {total['no_reference']} episode(s) have no (layout, route) in the "
              f"optimal json and are left out of normalized_path")

    if a.episodes:
        print()
        for r in rows:
            npath = fmt(r["normalized_path"], 7)
            print(f"  {r['task']:<46} {r['layout']:<18} "
                  f"task={str(r['task_success']):<5} "
                  f"coll_free={str(r['collision_free_success']):<5} "
                  f"planned={r['planned_path_len_m']} actual={r['executed_path_len_m']} "
                  f"[{r['path_length_source']}] norm={npath}")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"folder": str(root), "optimal": a.optimal,
                       "total": total, "per_ledger": per_ledger,
                       "episodes": rows}, fh, indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
