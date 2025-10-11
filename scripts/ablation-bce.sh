#!/bin/bash

video_modes=(0 1 2 3 4)

declare -A task_configs

task_configs["finger_spin"]="path_to_model/500000.pt"

cuda_idx=0

for video_mode in "${video_modes[@]}"; do
  for task_name in "${!task_configs[@]}"; do
    model_path="${task_configs[$task_name]}"
    domain_name="${task_name%_*}"

    echo "-------- Running task: ${task_name} with video_mode: ${video_mode} --------"

    CUDA_VISIBLE_DEVICES=${cuda_idx} xvfb-run -a python3 src/train-ftr.py \
        --log_dir runs-ablation-bce \
        --algorithm ftr_drq --seed 0 --cuda_idx ${cuda_idx} \
        --domain_name "${domain_name}" --task_name "${task_name#*_}" \
        --train_steps 200k \
        --supervised_steps 5k --sup_to_rl_warmup_steps 5k \
        --train_mode "video_hard_one_${video_mode}" --eval_mode "video_hard_one_${video_mode}" \
        --save_video --plot_selected --plot_segment \
        --use_selector --use_supervised --use_rl \
        --segment_interval 20 \
        --bce_loss \
        --pretrained_model_path "${model_path}"

    echo "-------- Finished running ${task_name}: ${video_mode} --------"
  done
done

echo "All tasks and video modes have been processed."