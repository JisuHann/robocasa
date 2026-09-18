# Rollout logging and post-evaluation

## 저장 구조

정책 worker는 rollout 중 episode 결과를 기록하고, post-evaluation은 ledger를
읽어 최종 지표를 계산합니다. SSI는 inference 중 계산하지 않습니다.

```
<output>/                         # 정책별 원본 rollout
  layout*/<task>/                 # run.log, trajectory_log.json, overview
  results*.json                   # worker 결과

<ledger>/                          # 평가 입력(ROBOCASA_LEDGER_DIR)
  run.json                         # model/policy/seed 메타데이터
  episodes.jsonl                   # episode별 성공·거리·접촉 요약
  traj/<episode>.npz               # pose, distance, contact, v/a/jerk series
  derived/rates.json               # 완료 episode 기준 누적 TSR/CSR
```

각 episode 종료 시 `task_success`, `collision_free_success`,
`contact_steps`/`collision_steps`, path length/time과 trajectory series가
ledger에 append됩니다. `derived/rates.json`은 지금까지 완료된 episode 전체를
기준으로 갱신됩니다.

## Post-evaluation entry point

계산은 `robocasa.metrics.summarize.summarize_post_evaluation`이 담당하고,
CLI는 다음 wrapper입니다.

```bash
python -m robocasa.scripts.summarize_post_evaluation_metrics \
  --inputs OUTPUT_A OUTPUT_B \
  --scope all task_success collision_free_task_success \
  --comparison individual \
  --out summary.json
```

`--inputs`에는 output 폴더 또는 ledger 폴더를 직접 줄 수 있습니다. output
폴더는 내부 ledger를 자동 탐색합니다.

### 필터 및 비교 옵션

- `--scope all`: 성공/실패 전체
- `--scope task_success`: task success episode만
- `--scope collision_free_task_success`: task 및 collision-free success만
- `--comparison individual`: 각 모델의 자체 eligible task 집합으로 계산
- `--comparison matched_intersection`: 선택한 scope를 통과한 공통
  `(layout, route, obstacle)`만 모델 간 비교
- `--allow-partial` (기본): H/M/L 중 2개 이상 tier가 있는 cell의 pair를 사용
- `--strict-complete`: H/M/L 세 tier가 모두 있는 cell만 사용
- `--optimal PATH`: optimal path JSON. 생략하거나 파일이 없으면 normalized
  path 지표는 `n/a`로 표시

SSI는 `(layout, route)` cell 안에서 obstacle을 H/M/L tier로 묶습니다.
`min_distance`는 trajectory 전체의 최소값 하나를 사용하고, `v/d`, `a/d`,
`J/d`는 `d_t ≤ 1.25m` timestep의 episode mean/max를 구한 뒤 tier 내부
obstacle 평균을 계산합니다. collision timestep의 분모 거리는 `ε=0.05m`로
바닥 처리합니다. 이후 cell별 Kendall tau와 H-M/M-L/H-L margin을 집계합니다.

## 여러 seed의 mean/std

동일 모델의 여러 seed ledger를 함께 넣고 `--aggregate-seeds`를 사용하면,
seed별 summary를 먼저 계산한 뒤 seed 간 `mean`, `std`, `n_seeds`를 출력합니다.
shard ledger는 seed별로 자동 그룹화됩니다.

```bash
python -m robocasa.scripts.summarize_post_evaluation_metrics \
  --inputs outputs/gemma4_seed0_vox_s*_ledger \
           outputs/gemma4_seed3_vox_s*_ledger \
           outputs/gemma4_seed16_vox_s*_ledger \
  --aggregate-seeds \
  --scope all task_success collision_free_task_success \
  --out gemma4_seed_summary.json
```

출력에는 seed별 원본 결과와 함께 TSR, CSR, normalized path, metric별 SSI
Kendall tau의 `mean/std/n_seeds`가 포함됩니다.

검증 helper는 `robocasa.metrics.summarize.check_post_evaluation`이며, 별도
inference-time SSI 계산이나 ledger worker는 필요하지 않습니다.
