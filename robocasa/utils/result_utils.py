"""Utilities for task registry queries, category parsing, and result saving."""

import os
import re
import json
import datetime

from robocasa.models.scenes.scene_registry import LayoutType

# Build layout name-to-id mapping
_LAYOUT_NAME_TO_ID = {
    lt.name: lt.value for lt in LayoutType if lt.value >= 0
}


def get_navigate_tasks():
    """Fetch all NavigateKitchen* task names from the robocasa registry."""
    import robocasa
    from robocasa.environments.kitchen.kitchen import REGISTERED_KITCHEN_ENVS

    return sorted(
        name for name in REGISTERED_KITCHEN_ENVS
        if name.startswith("NavigateKitchen")
        and name not in ("NavigateKitchenWithObstacles", "NavigateKitchen")
    )


def parse_task_spec(task_spec):
    """Parse TaskName_LAYOUT into (task_name, layout_id)."""
    for name, lid in sorted(_LAYOUT_NAME_TO_ID.items(), key=lambda x: -len(x[0])):
        suffix = f"_{name}"
        if task_spec.endswith(suffix):
            return (task_spec[:-len(suffix)], lid)
    return (task_spec, None)


def parse_task_categories(task_name):
    """Extract obstacle, raw safety-mode token, and route from a NavigateKitchen task name.

    The token is the raw string baked into the task class name (kept stable
    for asset/registry reasons); call sites translate it into the canonical
    `safety_mode` value used everywhere downstream.
    """
    m = re.match(
        r"NavigateKitchen(?P<obstacle>.+?)(?P<mode>NonBlocking|Blocking)Route(?P<route>[A-G])$",
        task_name,
    )
    if m:
        return (m.group("obstacle"), m.group("mode"), m.group("route"))
    return (None, None, None)


def save_results(results, summary, model, output_dir="results", filename=None, config=None):
    """Save per-task results and summary to a JSON file.

    `config` (optional) records the ablation parameters this run was launched
    with — without it, the same model name + timestamp can describe runs with
    very different obstacle_map_weight / prompt_variant / vlm_cameras settings.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename is None:
        model_safe = model.replace("/", "_") if model else "unknown_model"
        filename = f"{model_safe}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    output = {
        "model": model,
        "timestamp": timestamp,
        "config": config or {},
        "summary": summary,
        "results": results,
    }
    # Atomic write: a partial json on disk would crash the smoke/retry/analyze
    # watchers that poll this file every few seconds.
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(output, f, indent=2)
    os.replace(tmp, filepath)

    return filepath
