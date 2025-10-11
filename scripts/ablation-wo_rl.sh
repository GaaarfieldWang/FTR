#!/bin/bash

video_modes=(0 1 2 3 4)

declare -A task_configs

task_configs["walker_walk"]="path_to_model/500000.pt"
task_configs["walker_run"]="path_to_model/500000.pt"
task_configs["pendulum_swingup"]="path_to_model/500000.pt"
task_configs["hopper_stand"]="path_to_model/500000.pt"
task_configs["hopper_hop"]="path_to_model/500000.pt"
task_configs["franka_reach"]="path_to_model/500000.pt"
task_configs["finger_spin"]="path_to_model/500000.pt"
task_configs["cheetah_run"]="path_to_model/500000.pt"
task_configs["cartpole_swingup"]="path_to_model/500000.pt"
task_configs["franka_push"]="path_to_model/500000.pt"
task_configs["robosuite_Door"]="path_to_model/500000.pt"

for video_mode in "${video_modes[@]}"; do
  for task_name in "${!task_configs[@]}"; do
    model_path="${task_configs[$task_name]}"
    domain_name="${task_name%_*}"

    echo "-------- Running task: ${task_name} with video_mode: ${video_mode} --------"

    CUDA_VISIBLE_DEVICES=0 xvfb-run -a python3 src/train-ftr.py \
        --log_dir runs-ablation-wo-rl \
        --algorithm ftr_drq --seed 0 --cuda_idx 0 \
        --domain_name "${domain_name}" --task_name "${task_name#*_}" \
        --train_steps 200k \
        --supervised_steps 5k --sup_to_rl_warmup_steps 5k \
        --train_mode "video_hard_one_${video_mode}" --eval_mode "video_hard_one_${video_mode}" \
        --reward_factor 0.0 --inv_factor 0.0 --fwd_factor 0.0 \
        --save_video --plot_selected --plot_segment \
        --use_selector --use_supervised \
        --segment_interval 20 \
        --pretrained_model_path "${model_path}"

    echo "-------- Finished running ${task_name}: ${video_mode} --------"
  done
done

echo "All tasks and video modes have been processed."