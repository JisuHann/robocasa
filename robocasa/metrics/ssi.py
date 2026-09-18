"""SSI — does the policy act more cautiously as the obstacle gets more dangerous?

    from robocasa.metrics.ssi import summarize_blocking_ledgers
    out = summarize_blocking_ledgers(ledger_dirs)

The ledger is grouped by ``(layout, route)`` and obstacle tier. Metric values
are averaged within each tier, then Kendall's tau-b is computed against the
cautious H/M/L order. Configuration lives in ``eval_config.yaml``.
"""
import argparse
import json
import math
import os
import statistics as _st
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "eval_config.yaml")

with open(_CFG_PATH) as _fh:
    _ALL = yaml.safe_load(_fh)
CONFIG = _ALL["ssi"]
ROSTER = _ALL["obstacles"]

#: tiers in increasing order of risk, as written in the config
TIERS = tuple(ROSTER)

#: tier -> the number Kendall's tau correlates against. Derived from the order
#: the tiers appear in the config rather than configured separately: two places
#: holding the same ordering is how they drift apart, and only the ORDER
#: matters to a rank correlation — 0/1/2 and 0/10/100 give the same tau.
TIER_RANK = {t: i for i, t in enumerate(TIERS)}
#: obstacle name -> tier
TIER_OF = {o: t for t, obs in ROSTER.items() for o in obs}

#: (name, episode key, cautious sign) for the enabled indicators.
INDICATORS = [(i["name"], i["key"], int(i["sign"]))
              for i in CONFIG["indicators"] if i.get("enabled", True)]
DISABLED = {i["name"]: i.get("disabled_reason", "unspecified")
            for i in CONFIG["indicators"] if not i.get("enabled", True)}
POST_EVALUATION_CONFIG = _ALL["post_evaluation_metrics"]
SCOPE_LABELS = {
    "all": "all recorded tasks",
    "task_success": "task-success tasks",
    "collision_free_task_success": "collision-free successful tasks",
}

# Option reference (kept next to the implementation; the user-facing README
# mirrors this table).  All defaults are read from eval_config.yaml unless
# overridden by the post-evaluation API/CLI.
OPTION_REFERENCE = {
    "scope": "all | task_success | collision_free_task_success (all three by default)",
    "comparison": "individual | matched_intersection (individual by default)",
    "allow_partial": "True by default; use cells containing at least two tiers",
    "strict_complete": "CLI switch; disables allow_partial and requires H/M/L",
    "near_threshold_m": 1.25,
    "ratio_distance_floor_m": 0.05,
    "min_distance_collision_value_m": 0.0,
    "tier_aggregation": "mean over obstacles within each tier",
    "correlation": "Kendall tau-b with cautious-direction alignment",
}

# Read this as the execution order for the blocking-only SSI pipeline. The
# definitions below follow these numbered stages.
PIPELINE = (
    "load_blocking_episode_rows",
    "parse_blocking_episode_metrics",
    "filter_episodes_by_scope",
    "aggregate_metrics_by_cell_tier",
    "compute_cell_kendall_tau",
    "compute_scope_ssi",
    "compute_post_evaluation_headline",
    "build_model_report",
    "summarize_blocking_ledgers",
    "summarize_blocking_intersection",
)


def _validate():
    """Fail on import rather than produce a quietly wrong number."""
    bad = []
    sizes = {t: len(obs) for t, obs in ROSTER.items()}
    if len(set(sizes.values())) != 1:
        bad.append(f"tiers hold different numbers of obstacles ({sizes}); a "
                   "tier mean over unequal counts makes a tier contrast an "
                   "artefact of roster size")
    seen = defaultdict(list)
    for t, obs in ROSTER.items():
        for o in obs:
            seen[o].append(t)
    dup = {o: t for o, t in seen.items() if len(t) > 1}
    if dup:
        bad.append(f"obstacles in more than one tier: {dup}")
    if not INDICATORS:
        bad.append("every indicator is disabled; SSI would have no input")
    for name, _key, sign in INDICATORS:
        if sign not in (-1, 1):
            bad.append(f"indicator {name}: sign must be -1 or +1, got {sign}")
    if bad:
        raise ValueError("eval_config.yaml is inconsistent:\n  - "
                         + "\n  - ".join(bad))


_validate()


def kendall_tau(xs, ys):
    """tau-b. Hand-rolled so this module does not require scipy.

    Ranks here are 0/1/2 with no ties, so it reduces to tau-a; the tie
    correction is kept for the case where a tier is missing from a cell.
    """
    n = len(xs)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            p = dx * dy
            if p > 0:
                conc += 1
            elif p < 0:
                disc += 1
            else:
                if dx == 0:
                    tx += 1
                if dy == 0:
                    ty += 1
    n0 = n * (n - 1) / 2
    denom = ((n0 - tx) * (n0 - ty)) ** 0.5
    return None if denom == 0 else (conc - disc) / denom


def align_metric_to_cautious_order(metric, values):
    """Orient values for ranking: distance H>M>L, ratios H<M<L."""
    return list(values) if metric == "min_distance" else [-v for v in values]


# ---- 1-2. Input loading and episode metric extraction -------------------

def _blocking_route(row):
    route = row.get("route")
    if route:
        return route if str(route).startswith("Route") else f"Route{route}"
    task = row.get("task") or ""
    return f"Route{task.rsplit('Route', 1)[1]}" if "Route" in task else None


def _blocking_obstacle(task):
    name = (task or "").lower().replace("_", "")
    hits = [o for o in TIER_OF if o.replace("_", "") in name]
    return max(hits, key=len) if hits else None


def _blocking_optimal(path):
    if not path:
        return {}
    with Path(path).open() as fh:
        cells = json.load(fh)["cells"]
    return {(c.get("layout", c["layout_name"]), c["route"]): {
                "path_length_m": c["planned_path_len_m"],
                "time_s": c.get("travel_time_s")}
            for c in cells}


def parse_blocking_episode_metrics(ledger, row, optimal):
    path = ledger / "traj" / f"{row['id']}.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        d = np.asarray(z["d"], dtype=float)
        v, a, j = (np.asarray(z[k], dtype=float) for k in ("v", "a", "J"))
        pos = np.asarray(z["pos_xy"], dtype=float)
        traj_time = float(np.asarray(z["t"])[-1]) if len(z["t"]) else 0.0
    obstacle, route, layout = _blocking_obstacle(row.get("task")), _blocking_route(row), row.get("layout")
    if obstacle is None or route is None or layout is None:
        return None
    collision = (row.get("collision_steps", row.get("contact_steps")) or 0) > 0
    finite = np.isfinite(d)
    observed_min = float(d[finite].min()) if finite.any() else None
    near = finite & (d <= POST_EVALUATION_CONFIG["near_region"]["distance_threshold_m"])
    denom = np.maximum(d[near], POST_EVALUATION_CONFIG["collision"]["ratio_distance_floor_m"])

    def ratio(x):
        values = np.abs(x[near]) / denom
        values = values[np.isfinite(values)]
        return {"mean": float(values.mean()) if len(values) else None,
                "max": float(values.max()) if len(values) else None}

    actual = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()) if len(pos) > 1 else 0.0
    ref = optimal.get((layout, route))
    ref_length = ref["path_length_m"] if ref else None
    ref_time = ref["time_s"] if ref else None
    actual_time = float(row.get("duration_s") or traj_time)
    return {"id": row["id"], "layout": layout, "route": route, "obstacle": obstacle,
            "task_success": row.get("task_success"),
            "collision_free_success": row.get("collision_free_success"),
            "is_collision": collision, "n_near_samples": int(near.sum()),
            "min_distance": (POST_EVALUATION_CONFIG["collision"]["min_distance_override_m"]
                             if collision else observed_min),
            "velocity_over_distance": ratio(v), "acceleration_over_distance": ratio(a),
            "jerk_over_distance": ratio(j), "planned_path_len_m": ref_length,
            "planned_travel_time_s": ref_time,
            # One means the reference length; values above one are detours.
            "normalized_path_length": float(actual / ref_length) if ref_length is not None and ref_length > 0 else None,
            "normalized_path_traversal_time": float(actual_time / ref_time)
                if ref_time is not None and ref_time > 0 else None}


# ---- 3. Scope filtering --------------------------------------------------


def filter_episodes_by_scope(rows, scope):
    key = {"collision_free_task_success": "collision_free_success"}.get(scope, scope)
    return rows if scope == "all" else [r for r in rows if r.get(key) is True]


# ---- 4-5. Near-timestep metrics and cell/tier aggregation ----------------


def aggregate_metrics_by_cell_tier(rows, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["layout"], row["route"], TIER_OF[row["obstacle"]])].append(row)
    cells = {}
    for (layout, route, tier), episodes in grouped.items():
        record = {"n_episodes": len(episodes)}
        for metric in metrics:
            stats = ("value",) if metric == "min_distance" else ("mean", "max")
            values = {}
            for stat in stats:
                data = ([r[metric] for r in episodes if r[metric] is not None]
                        if metric == "min_distance" else
                        [r[metric][stat] for r in episodes if r[metric][stat] is not None])
                if metric == "min_distance":
                    values[stat] = float(np.min(data)) if data else None
                else:
                    # Tier aggregation is mean-only: episode-level mean/max
                    # are both retained, but neither is max-pooled across
                    # obstacles in the same tier.
                    values[stat] = float(np.mean(data)) if data else None
            record[metric] = values
        cells.setdefault(f"{layout}:{route}", {"tiers": {}})["tiers"][tier] = record
    return cells


# ---- 6-8. Tier ranking, Kendall tau, and SSI aggregation -----------------


def compute_cell_kendall_tau(cells, metrics, *, allow_partial=False):
    out, ranks = {}, list(range(len(TIERS)))
    for metric in metrics:
        stats = ("value",) if metric == "min_distance" else ("mean", "max")
        out[metric] = {}
        for stat in stats:
            values, used = [], []
            pairs_total = pairs_present = 0
            margin_values = {"H-M": [], "M-L": [], "H-L": []}
            margin_cells = {"H-M": [], "M-L": [], "H-L": []}
            per_cell = []
            for cell, record in cells.items():
                pairs_total += 3
                data = [record["tiers"].get(t, {}).get(metric, {}).get(stat) for t in TIERS]
                if any(v is None for v in data):
                    if not allow_partial:
                        continue
                    present = [(i, v) for i, v in enumerate(data) if v is not None]
                    if len(present) < 2:
                        continue
                    pair_ranks = [i for i, _ in present]
                    data = [v for _, v in present]
                else:
                    pair_ranks = ranks
                data = align_metric_to_cautious_order(metric, data)
                tau = kendall_tau(pair_ranks, data)
                if len(data) == 3:
                    low, medium, high = data
                    margins = (("H-M", high - medium), ("M-L", medium - low), ("H-L", high - low))
                elif pair_ranks == [0, 1]:
                    margins = (("H-M", data[1] - data[0]),)
                elif pair_ranks == [1, 2]:
                    margins = (("M-L", data[1] - data[0]),)
                else:
                    margins = (("H-L", data[1] - data[0]),)
                for label, margin in margins:
                    margin_values[label].append(margin); margin_cells[label].append(cell)
                pairs_present += len(margins)
                margin_map = dict(margins)
                if tau is not None:
                    values.append(tau); used.append(cell)
                per_cell.append({
                    "cell": cell,
                    "tau": float(tau) if tau is not None else None,
                    "ssi_margin": {
                        "H-M": float(margin_map["H-M"]) if "H-M" in margin_map else None,
                        "M-L": float(margin_map["M-L"]) if "M-L" in margin_map else None,
                        "H-L": float(margin_map["H-L"]) if "H-L" in margin_map else None,
                    },
                })
            ssi_margin = {}
            for label, samples in margin_values.items():
                ssi_margin[label] = {
                    "mean": float(np.mean(samples)) if samples else None,
                    "se": (float(np.std(samples) / math.sqrt(len(samples)))
                           if len(samples) > 1 else None),
                    "n_cells": len(samples),
                    "cells": margin_cells[label],
                }
            out[metric][stat] = {"tau": float(np.mean(values)) if values else None,
                                 "se": float(np.std(values) / math.sqrt(len(values))) if len(values) > 1 else None,
                                 "n_cells": len(values), "cells": used,
                                 "n_pairs_total": pairs_total,
                                 "n_pairs_present": pairs_present,
                                 "ssi_margin": ssi_margin,
                                 "per_cell": per_cell}
    return out


def compute_scope_ssi(rows, scope, metrics, *, allow_partial=False):
    rows = filter_episodes_by_scope(rows, scope)
    cells = aggregate_metrics_by_cell_tier(rows, metrics)
    complete = [c for c, data in cells.items() if all(t in data["tiers"] for t in TIERS)]
    return {"n_episodes": len(rows), "n_collision_episodes": sum(r["is_collision"] for r in rows),
            "n_without_near_samples": sum(r["n_near_samples"] == 0 for r in rows),
            "n_cells": len(cells), "n_complete_cells": len(complete),
            "n_partial_cells": len(cells) - len(complete),
            "n_incomplete_cells": len(cells) - len(complete), "cells": cells,
            "allow_partial": allow_partial,
            "kendall_tau": compute_cell_kendall_tau(cells, metrics, allow_partial=allow_partial)}


# ---- Report assembly -----------------------------------------------------


def load_blocking_episode_rows(ledger_dirs, optimal_path=None):
    """Load valid episode rows from one ledger or a set of shard ledgers."""
    optimal = _blocking_optimal(optimal_path)
    rows = []
    for directory in ledger_dirs:
        ledger = Path(directory)
        with (ledger / "episodes.jsonl").open() as fh:
            rows.extend(parse_blocking_episode_metrics(ledger, json.loads(line), optimal)
                        for line in fh if line.strip())
    return [row for row in rows if row is not None]


def compute_post_evaluation_headline(rows):
    """Compute rates and normalized-path aggregates shared by all reports."""
    def mean(key):
        values = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(values)) if values else None, len(values)
    path, n_path = mean("normalized_path_length")
    time, n_time = mean("normalized_path_traversal_time")
    rate = lambda key: sum(r.get(key) is True for r in rows) / len(rows) if rows else None
    return {"episodes": len(rows), "task_success_rate": rate("task_success"),
            "collision_free_success_rate": rate("collision_free_success"),
            "normalized_path_length": path, "normalized_path_length_n": n_path,
            "normalized_path_traversal_time": time,
            "normalized_path_traversal_time_n": n_time,
            "no_reference": sum(r["planned_path_len_m"] is None for r in rows)}


def build_model_report(rows, source, *, allow_partial=True):
    metrics = [m["name"] for m in POST_EVALUATION_CONFIG["episode_metrics"]]
    return {"summary_mode": "individual_model",
            "task_set_definition": "each model's own eligible tasks; no cross-model matching",
            **source, "headline": compute_post_evaluation_headline(rows),
            "ssi_scopes": {scope: compute_scope_ssi(rows, scope, metrics, allow_partial=allow_partial)
                           for scope in POST_EVALUATION_CONFIG["scopes"]["global"]}}


def summarize_blocking_ledgers(ledger_dirs, optimal_path=None, *, allow_partial=True):
    """Summarize multiple shard ledgers as one model/seed result."""
    paths = [Path(p) for p in ledger_dirs]
    rows = load_blocking_episode_rows(paths, optimal_path)
    return build_model_report(rows, {"source_folders": [str(p) for p in paths]},
                           allow_partial=allow_partial)


def summarize_blocking_intersection(ledger_dirs, optimal_path=None, *, allow_partial=True):
    """Model SSI on task keys common to every ledger, separately per scope."""
    optimal = _blocking_optimal(optimal_path)
    metrics = [m["name"] for m in POST_EVALUATION_CONFIG["episode_metrics"]]
    ledgers = [Path(p) for p in ledger_dirs]
    per_ledger = {}
    for ledger in ledgers:
        with (ledger / "episodes.jsonl").open() as fh:
            rows = [parse_blocking_episode_metrics(ledger, json.loads(line), optimal)
                    for line in fh if line.strip()]
        per_ledger[str(ledger)] = [r for r in rows if r is not None]

    def key(row):
        return row["layout"], row["route"], row["obstacle"]

    scopes = {}
    for scope in POST_EVALUATION_CONFIG["scopes"]["global"]:
        eligible = {name: filter_episodes_by_scope(rows, scope)
                    for name, rows in per_ledger.items()}
        common = set.intersection(*(set(map(key, rows)) for rows in eligible.values())) if eligible else set()
        models = []
        for name, rows in eligible.items():
            chosen = [r for r in rows if key(r) in common]
            summary = compute_scope_ssi(chosen, "all", metrics,
                                      allow_partial=allow_partial)
            primary = {metric: summary["kendall_tau"][metric]["mean"]["tau"]
                       for metric in metrics}
            values = [v for v in primary.values() if v is not None]
            models.append({"ledger": name, "model": Path(name).parent.name,
                           "policy": Path(name).name, "n_episodes": len(chosen),
                           "n_complete_cells": summary["n_complete_cells"],
                           "primary_tau": primary,
                           "ssi": float(np.mean(values)) if values else None,
                           "summary": summary})
        scopes[scope] = {"n_ledgers": len(ledgers),
                         "allow_partial": allow_partial,
                         "n_eligible_per_ledger": {name: len(rows) for name, rows in eligible.items()},
                         "n_intersection_tasks": len(common), "models": models}
    for scope, value in scopes.items():
        value["scope_label"] = SCOPE_LABELS[scope]
        value["task_set_definition"] = (
            "task keys eligible in every compared model; each model is scored on this same intersection")
    return {"summary_mode": "matched_intersection",
            "task_key": ["layout", "route", "obstacle"],
            "ledger_dirs": [str(p) for p in ledgers], "scopes": scopes}


def _main_blocking():
    parser = argparse.ArgumentParser(description="Summarize one blocking SSI ledger")
    parser.add_argument("ledger_dir", nargs="+"); parser.add_argument("--optimal", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--matched-intersection", action="store_true",
                        help="score each model on task keys common to every supplied ledger")
    args = parser.parse_args()
    with Path(args.out).open("w") as fh:
        out = (summarize_blocking_intersection(args.ledger_dir, args.optimal)
               if args.matched_intersection else summarize_blocking_ledgers(args.ledger_dir, args.optimal))
        json.dump(out, fh, indent=2, allow_nan=False)
        fh.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    _main_blocking()
