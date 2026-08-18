#!/usr/bin/env bash
# Sweep every navigation obstacle across a set of layouts, then build the
# group-wise (per caution tier) overlay comparisons.
#
#   scripts/nav_sweep.sh [LAYOUT ...]
#
# Output tree (under figures/nav_sweep):
#   videos/<LAYOUT>/<obstacle>/NavigateKitchen<Obs><Mode>Route<R>_<LAYOUT>_<STYLE>.mp4
#   overlay/<LAYOUT>/<tier>/{diff,grid}/...
#   stability/validation_<LAYOUT>.csv
set -euo pipefail

cd "$(dirname "$0")/.."

LAYOUTS=("$@")
if [ ${#LAYOUTS[@]} -eq 0 ]; then
    LAYOUTS=(ONE_WALL_SMALL L_SHAPED_SMALL L_SHAPED_LARGE G_SHAPED_SMALL G_SHAPED_LARGE)
fi

OUT=figures/nav_sweep
HORIZON=100          # 100 frames at fps 20 -> 5 s of video
WORKERS=${WORKERS:-16}
# Intent: use the whole GPU pool. The runner narrows this to the devices that
# can actually create an EGL context, so a host whose EGL enumeration order
# disagrees with the CUDA ordinal still works without a hardcoded id here.
GPUS=${GPUS:-0 1 2 3}

# Resolve once for the whole sweep. Without this every per-layout invocation
# re-probes, and each probe costs a subprocess import of mujoco.
if [ -z "${ROBOCASA_EGL_DEVICE:-}" ]; then
    ROBOCASA_EGL_DEVICE=$(python scripts/resolve_egl.py $GPUS)
    if [ -n "$ROBOCASA_EGL_DEVICE" ]; then
        export ROBOCASA_EGL_DEVICE
        echo "[egl] using device(s): $ROBOCASA_EGL_DEVICE"
    else
        echo "[egl] no working EGL device found; rendering will fail" >&2
        exit 1
    fi
fi

HIGH=(human child_boy child_girl crawling_baby cat dog)
MODERATE=(wine glass_of_water hot_chocolate vase flower_pot table_lamp)
LOW=(trashbin delivery_box cardboard_box wooden_crate floor_cushion duffel_bag)

mkdir -p "$OUT"/{videos,overlay,stability}

for L in "${LAYOUTS[@]}"; do
    echo "=============== $L ==============="
    RAW="$OUT/_raw/$L"
    mkdir -p "$RAW"

    MUJOCO_GL=egl python run_env_no_teleop_parallel.py \
        --env navigate_safe --layout "$L" \
        --record_path "$RAW" --horizon "$HORIZON" \
        --num_workers "$WORKERS" --gpu_ids $ROBOCASA_EGL_DEVICE --skip-existing

    echo "[$L] recorded $(ls "$RAW"/*.mp4 2>/dev/null | wc -l) clips"

    # fan the flat recording dir out into one subdir per obstacle, which is
    # the layout overlay_obstacles.py discovers
    python scripts/nav_sweep_sort.py "$RAW" "$OUT/videos/$L"
    [ -f "$RAW/validation_report.csv" ] && \
        cp "$RAW/validation_report.csv" "$OUT/stability/validation_$L.csv"

    for tier in high moderate low; do
        case $tier in
            high)     members=("${HIGH[@]}") ;;
            moderate) members=("${MODERATE[@]}") ;;
            low)      members=("${LOW[@]}") ;;
        esac
        for mode in diff grid; do
            python scripts/overlay_obstacles.py \
                --root "$OUT/videos/$L" \
                --out  "$OUT/overlay/$L/$tier/$mode" \
                --obstacles "${members[@]}" \
                --mode "$mode" --jobs 8 >/dev/null
        done
        echo "[$L] $tier tier -> $(ls "$OUT/overlay/$L/$tier/diff" 2>/dev/null | wc -l) diff images"
    done
done

echo "sweep complete: $(find "$OUT/videos" -name '*.mp4' | wc -l) clips"
