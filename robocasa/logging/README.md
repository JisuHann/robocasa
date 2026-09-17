# Rollout logging and post-evaluation

Each rollout records its own detailed artifacts under the policy's `runs/`
directory. For a multi-worker sweep, every completed episode is also appended
to one shared RoboCasa ledger:

```
<ledger>/<tag>/<policy>/
  run.json             # sweep metadata
  episodes.jsonl       # one task/collision-free verdict per episode
  traj/<episode>.npz   # pose, distance, contact, v/a/jerk trajectory
  derived/rates.json   # cumulative success rates
```

`runs/` is for policy-specific rollout inspection; the ledger is the
worker-independent input to RoboCasa evaluation.

- At each episode end, the rollout supplies and records `task_success` and
  `collision_free_success`; `derived/rates.json` is refreshed over all finished
  episodes.
- After the sweep, `robocasa.metrics.summarize.summarize_post_evaluation`
  is the single post-evaluation entry point for rates, normalized path, and
  SSI. The `robocasa.scripts.summarize_post_evaluation_metrics` command is
  only its CLI wrapper; `utils.check_post_evaluation_metrics` validates the
  returned report. Use `--matched-intersection` when comparing models on the
  same task intersection. SSI is never computed during inference.
