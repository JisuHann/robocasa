"""Cross-check the four tables that define an obstacle, and fail loudly.

An obstacle is defined in four places that nothing keeps in sync:

    _OBSTACLE_CLASS_NAMES      registers it and generates its task classes
    OBSTACLE_BOUNDARY_RADIUS   the r_b the env scores boundary intrusion at
    TIER_OF                    the caution tier SSI aggregates by
    TIER_R_B                   the r_b SSI reports per tier

Every mismatch found so far has failed SILENTLY rather than raising:

  * missing from OBSTACLE_BOUNDARY_RADIUS -> scored at the 0.5 m class
    default, which matches no tier, so the obstacle is neither High nor
    Medium nor Low but something in between.
  * missing from TIER_OF -> `compute_ssi` does `if ell is None: continue`,
    so every episode of that obstacle runs, burns GPU, and is then dropped
    from the metric with no warning. `Kettlebell` sits here today: 20 task
    classes, 160 instances at 8 layouts.
  * r_b disagreeing between the env and SSI -> the env scores against one
    radius while the paper reports another.

Also checks the spawn class, which is a separate silent hazard: an obstacle
in neither TABLE_OBSTACLES nor TIPPY_FLOOR_OBSTACLES defaults to a 5 cm
floor drop, and tall / high-CoM / round meshes topple on that impact. That
is not detectable here (it needs a physics rollout), so this script only
reports the classification for review and points at the sweep.

Exit status: 0 clean, 1 if any inconsistency was found — usable in CI.

    python validation/check_tier_consistency.py
    python validation/check_tier_consistency.py --quiet   # only problems
"""
import argparse
import sys
from collections import defaultdict

TIER_ORDER = ("High", "Medium", "Low")


def load_tables():
    from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import (
        OBSTACLE_BOUNDARY_RADIUS, _DEFAULT_BOUNDARY_RADIUS,
        _OBSTACLE_CLASS_NAMES, TABLE_OBSTACLES, TIPPY_FLOOR_OBSTACLES,
    )
    from robocasa.utils.ssi import TIER_OF, TIER_R_B
    return dict(
        radius=OBSTACLE_BOUNDARY_RADIUS, default_radius=_DEFAULT_BOUNDARY_RADIUS,
        classes=_OBSTACLE_CLASS_NAMES, table=TABLE_OBSTACLES,
        tippy=TIPPY_FLOOR_OBSTACLES, tier_of=TIER_OF, tier_rb=TIER_R_B,
    )


def spawn_class(key, t):
    if key in t["table"]:
        return "table (+1 cm)"
    if key in t["tippy"]:
        return "tippy (+2 cm)"
    return "plain floor (+5 cm)"


def check(quiet=False):
    t = load_tables()
    problems = []
    by_tier = defaultdict(list)

    rows = []
    for key, cls in sorted(t["classes"].items()):
        rb_env = t["radius"].get(key)
        tier = t["tier_of"].get(cls)
        rb_ssi = t["tier_rb"].get(cls)
        issues = []
        if rb_env is None:
            issues.append(f"no OBSTACLE_BOUNDARY_RADIUS entry -> scored at the "
                          f"{t['default_radius']} m default")
        if tier is None:
            issues.append("no TIER_OF entry -> every episode is dropped from SSI")
        if rb_ssi is None:
            issues.append("no TIER_R_B entry")
        if rb_env is not None and rb_ssi is not None and abs(rb_env - rb_ssi) > 1e-9:
            issues.append(f"r_b disagrees: env={rb_env} ssi={rb_ssi}")
        if issues:
            problems.append((key, cls, issues))
        by_tier[tier].append(key)
        rows.append((key, cls, rb_env, tier, rb_ssi, spawn_class(key, t), issues))

    # tables naming obstacles that are not registered
    registered = set(t["classes"].values())
    for name in sorted(set(t["tier_of"]) | set(t["tier_rb"])):
        if name not in registered:
            # a retired alias is harmless; flag it as info, not a failure
            if not quiet:
                print(f"[info] SSI tables list {name!r}, which is not a "
                      f"registered obstacle (stale alias?)")
    for key in sorted(t["radius"]):
        if key not in t["classes"]:
            problems.append((key, "-", ["in OBSTACLE_BOUNDARY_RADIUS but not "
                                        "registered in _OBSTACLE_CLASS_NAMES"]))

    if not quiet:
        hdr = (f"{'obstacle':17s} {'ClassName':15s} {'r_b(env)':>9s} "
               f"{'tier':>7s} {'r_b(ssi)':>9s}  {'spawn':18s}")
        print(hdr)
        print("-" * len(hdr))
        for key, cls, rb_env, tier, rb_ssi, spawn, issues in rows:
            flag = "  <-- " + "; ".join(issues) if issues else ""
            print(f"{key:17s} {cls:15s} {str(rb_env):>9s} {str(tier):>7s} "
                  f"{str(rb_ssi):>9s}  {spawn:18s}{flag}")

        print("\ntier balance:")
        for tier in TIER_ORDER:
            members = by_tier.get(tier, [])
            print(f"  {tier:7s} {len(members):2d}  {', '.join(members)}")
        untiered = by_tier.get(None, [])
        if untiered:
            print(f"  {'(none)':7s} {len(untiered):2d}  {', '.join(untiered)}")

        sizes = {tier: len(by_tier.get(tier, [])) for tier in TIER_ORDER}
        if len(set(sizes.values())) != 1:
            print(f"\n[warn] tiers are unbalanced ({sizes}) — a per-tier mean "
                  f"is then computed over different numbers of obstacle types, "
                  f"so a tier contrast can be driven by roster size.")

        print("\nSpawn class is NOT verified here — it needs a physics rollout. "
              "After any roster change run:\n"
              "  python validation/check_obstacle_stability.py --seeds 0\n"
              "and read `starts_fallen` as 'wrong spawn class'.")

    if problems:
        print(f"\nFAIL — {len(problems)} inconsistency(ies):")
        for key, cls, issues in problems:
            for i in issues:
                print(f"  {key} ({cls}): {i}")
        return 1
    print("\nOK — all registered obstacles agree across radius, tier and SSI tables.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true",
                    help="print only problems (for CI)")
    args = ap.parse_args()
    sys.exit(check(quiet=args.quiet))


if __name__ == "__main__":
    main()
