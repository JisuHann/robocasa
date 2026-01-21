#!/bin/bash

# find all glb files in the 3D_assets directory and convert them using the import_glb_model.py script
#!/usr/bin/env bash
set -e

# ====== 설정 ======
ASSETS_DIR="/Users/hyunw3/robotics/robotics-safety/benchmark/robocasa/3D_assets"
OUTPUT_ROOT="./3D_converted_models"
SCRIPT_PATH="robocasa/scripts/model_zoo/import_glb_model.py"

# ====== 준비 ======
mkdir -p "$OUTPUT_ROOT"

# ====== GLB 파일 순회 ======
find "$ASSETS_DIR" -type f -name "*.glb" | while read -r glb_path; do
    # 파일명 (확장자 제거)
    base_name=$(basename "$glb_path" .glb)

    # 출력 경로
    output_path="$OUTPUT_ROOT/$base_name"
    mkdir -p "$output_path"

    echo "================================================="
    echo "Converting: $glb_path"
    echo "Output dir: $output_path"

    python "$SCRIPT_PATH" \
        --prescale \
        --center \
        --no_cached_coll \
        --path "$glb_path" \
        --output_path "$output_path"

    echo "Done: $base_name"
done

echo "✅ All GLB files converted."
# previous one by one
# input_path="/Users/hyunw3/robotics/robotics-safety/benchmark/robocasa/3D_assets/coffee.glb"
# output_path="./3D_converted_models/coffee"
# mkdir -p "$output_path"
# python robocasa/scripts/model_zoo/import_glb_model.py --prescale --center --no_cached_coll --path "$input_path"mj --output_path "$output_path"