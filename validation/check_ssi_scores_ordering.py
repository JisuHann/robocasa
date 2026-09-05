"""The new SSI must compute on real results, not just import.

The last SSI test passed while the code it was meant to cover never ran: its
fixture lacked the fields the axis path reads, so the function returned None and
the NameError inside it went unseen. This one asserts the number exists, that
the pairing kept episodes, and that a deliberately caution-ordered input scores
+1 while its reverse scores -1.
"""
import sys

from robocasa.metrics import ssi

TIERS = ("low", "medium", "high")
OBS = {t: ssi.CONFIG["tiers"][t] for t in TIERS}


def episode(layout, route, obstacle, mode, value):
    """One episode, with each indicator set so `value` means "more cautious".

    The indicators do not all point the same way: speed, acceleration and jerk
    are cautious when SMALL (sign -1) while distance is cautious when LARGE
    (sign +1). Writing one number into every field would therefore score the
    distance axes backwards and the test would be measuring its own fixture.
    """
    name = (f"NavigateKitchen{obstacle.title().replace('_', '')}"
            f"{'Blocking' if mode == 'blocking' else 'NonBlocking'}Route{route}")
    ev = {"task_success": True, "success": True}
    for _n, key, sign, _c in ssi.INDICATORS:
        # sign -1: smaller is more cautious, so a cautious episode gets the
        # smaller number. sign +1: the reverse.
        ev[key] = value if sign < 0 else 2.0 - value
    return {"task_info": {"task_name": name, "layout_id": layout}, "evaluation": ev}


def build(order):
    """order maps tier -> how cautious that tier's blocking episode is.

    Lower means more cautious. episode() turns that into the right direction
    for each indicator's sign; the nonblocking baseline is always 1.0, the
    neutral value.
    """
    out = []
    for layout in (0, 2):
        for route in ("A", "B"):
            for tier in TIERS:
                for obstacle in OBS[tier]:
                    out.append(episode(layout, route, obstacle, "blocking",
                                       order[tier]))
                    out.append(episode(layout, route, obstacle, "nonblocking",
                                       1.0))
    return out


checks = {}

# Caution rises with tier: the riskier the obstacle, the smaller the value.
aligned = ssi.compute(build({"low": 1.0, "medium": 0.8, "high": 0.6}))
print("aligned   ssi =", aligned["ssi"], " pairs used =",
      aligned["ssi_n_pairs_used"], "/", aligned["ssi_n_pairs"])
checks["aligned scores +1"] = aligned["ssi"] == 1.0
checks["pairing kept every episode"] = (
    aligned["ssi_n_pairs_used"] == aligned["ssi_n_pairs"] > 0)
checks["every indicator reported"] = (
    set(aligned["ssi_per_indicator"]) == {n for n, _k, _s, _c in ssi.INDICATORS})

# Exactly inverted.
inverted = ssi.compute(build({"low": 0.6, "medium": 0.8, "high": 1.0}))
print("inverted  ssi =", inverted["ssi"])
checks["inverted scores -1"] = inverted["ssi"] == -1.0

# Exactly equal tier means leave tau undefined, not zero: a correlation needs
# both variables to vary. Reporting None is the honest answer, and it is what a
# degenerate synthetic input deserves.
flat = ssi.compute(build({"low": 0.8, "medium": 0.8, "high": 0.8}))
print("flat      ssi =", flat["ssi"], "(undefined, as it should be)")
checks["exactly flat is undefined"] = flat["ssi"] is None

# The realistic no-signal case: tier means differ, but not in tier order. Half
# the cells rank one way and half the other, so the mean lands at chance — which
# is 0 here, where the old binary-indicator form put it at 0.5.
mixed = ssi.compute(
    build({"low": 1.0, "medium": 0.8, "high": 0.6})
    + [dict(e, task_info=dict(e["task_info"],
                              layout_id=e["task_info"]["layout_id"] + 10))
       for e in build({"low": 0.6, "medium": 0.8, "high": 1.0})])
print("mixed     ssi =", mixed["ssi"])
checks["mixed lands at chance (0)"] = abs(mixed["ssi"]) < 1e-9

# An empty input must say so rather than inventing a number.
empty = ssi.compute([])
checks["empty gives None"] = empty["ssi"] is None

print()
for k, v in checks.items():
    print(f"  {'OK  ' if v else 'FAIL'} {k}")
print("RESULT:", "PASS" if all(checks.values()) else "FAIL")
sys.exit(0 if all(checks.values()) else 1)
