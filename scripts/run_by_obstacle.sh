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
)
target_folder=./test_video
mkdir -p ${target_folder}
for obstacle in "${obstacle_list[@]}"; do
    python run_env_no_teleop_parallel.py  --layout all --env navigate_safe\
        --num_workers 16 --gpu_ids 0 1 2 3 --horizon 15 \
        --record_path=${target_folder}/${obstacle} --filter_env_keyword=${obstacle} | tee -a ${target_folder}/log_${obstacle}.txt
done