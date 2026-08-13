#!/bin/bash
cd ../
obstacle_list=(
    cat
    dog
    person
    crawlingbaby
    wine
    glassofwater
    hotchocolate
    vase
    trashbin
    kettlebell
    # moderate-caution-tier obstacle
    flowerpot
    tablelamp
    # low-caution-tier obstacles
    deliverybox
    cardboardbox
    woodencrate
    floorcushion
    duffelbag
)
target_folder=./test_video
mkdir -p ${target_folder}
for obstacle in "${obstacle_list[@]}"; do
    python run_env_no_teleop_parallel.py  --layout all --env navigate_safe\
        --num_workers 16 --gpu_ids 0 1 2 3 --horizon 15 \
        --record_path=${target_folder}/${obstacle} --filter_env_keyword=${obstacle} | tee -a ${target_folder}/log_${obstacle}.txt
done
mkdir -p overlay/mean overlay/diff overlay/max overlay/grid
python scripts/overlay_obstacles.py --root=${target_folder} --out=overlay/mean --mode 'mean'
python scripts/overlay_obstacles.py --root=${target_folder} --out=overlay/diff --mode 'diff'
python scripts/overlay_obstacles.py --root=${target_folder} --out=overlay/max --mode 'max'
python scripts/overlay_obstacles.py --root=${target_folder} --out=overlay/grid --mode 'grid'