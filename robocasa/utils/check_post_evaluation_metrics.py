"""Contract test for post-evaluation ledger metrics.

It catches two scoring mistakes that are easy to hide in aggregate output:
contact_steps must force min-distance to zero, and a failed/colliding task
must remain in the `all` scope while being excluded from task_success.
"""
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np


MODULE = Path(__file__).parents[1] / "metrics" / "ssi.py"
spec = importlib.util.spec_from_file_location("unpaired_ssi", MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def write_episode(root, *, name, obstacle, success, collision_free, contacts,
                  distance, velocity):
    episode_id = f"l0_{name}"
    np.savez(root / "traj" / f"{episode_id}.npz",
             t=np.array([0.0, 0.05]),
             pos_xy=np.array([[0.0, 0.0], [1.0, 0.0]]),
             yaw=np.zeros(2), d=np.asarray(distance),
             v=np.asarray(velocity), a=np.ones(2), J=np.ones(2),
             in_contact=np.zeros(2, dtype=bool))
    return {
        "id": episode_id, "task": f"NavigateKitchen{obstacle}BlockingRouteA",
        "layout": 0, "route": "A", "task_success": success,
        "collision_free_success": collision_free, "contact_steps": contacts,
    }


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "traj").mkdir()
    rows = [
        write_episode(root, name="low", obstacle="Trashbin", success=True,
                      collision_free=True, contacts=0,
                      distance=[0.5, 0.5], velocity=[0.2, 0.2]),
        write_episode(root, name="medium", obstacle="Wine", success=True,
                      collision_free=True, contacts=0,
                      distance=[0.4, 0.4], velocity=[0.2, 0.2]),
        write_episode(root, name="high", obstacle="Human", success=False,
                      collision_free=False, contacts=2,
                      distance=[-0.1, 0.2], velocity=[0.2, 0.2]),
    ]
    (root / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    optimal = root / "optimal.json"
    optimal.write_text(json.dumps({"cells": [
        {"layout": 0, "layout_name": "fixture", "route": "RouteA",
         "planned_path_len_m": 2.0}]}))

    out = mod.summarize_unpaired_ledger(root, optimal)
    intersection = mod.summarize_blocking_intersection([root], optimal)

checks = {
    "rates use every recorded task": (
        out["headline"]["episodes"] == 3
        and out["headline"]["task_success_rate"] == 2 / 3
        and out["headline"]["collision_free_success_rate"] == 2 / 3),
    "normalized path is actual path over A-star reference": (
        out["headline"]["normalized_path"] == 0.5
        and out["headline"]["normalized_path_n"] == 3),
    "all scope keeps colliding failed episode and zeroes d_min": (
        out["ssi_scopes"]["all"]["n_episodes"] == 3
        and out["ssi_scopes"]["all"]["cells"]["0:RouteA"]["tiers"]
        ["high"]["min_distance"]["mean"] == 0.0),
    "success scope excludes failed episode and rejects incomplete tier cell": (
        out["ssi_scopes"]["task_success"]["n_episodes"] == 2
        and out["ssi_scopes"]["task_success"]["n_complete_cells"] == 0),
    "collision-free scope maps to the ledger outcome field": (
        out["ssi_scopes"]["collision_free_task_success"]["n_episodes"] == 2),
    "ratio denominator floors negative collision distance": (
        out["ssi_scopes"]["all"]["cells"]["0:RouteA"]["tiers"]
        ["high"]["velocity_over_distance"]["max"] == 4.0),
    "intersection summary returns one model on its common safe-success tasks": (
        intersection["scopes"]["all"]["n_intersection_tasks"] == 3
        and intersection["scopes"]["task_success"]["n_intersection_tasks"] == 2
        and intersection["scopes"]["collision_free_task_success"]["n_intersection_tasks"] == 2
        and len(intersection["scopes"]["all"]["models"]) == 1
        and intersection["summary_mode"] == "matched_intersection"),
}

for name, ok in checks.items():
    print(f"{'OK' if ok else 'FAIL'}  {name}")
raise SystemExit(0 if all(checks.values()) else 1)
