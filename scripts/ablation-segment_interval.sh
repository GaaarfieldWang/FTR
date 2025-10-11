#!/bin/bash

# Run finger spin with different segment intervals. Our default of segment interval is 20.

video_modes=(0 1 2 3 4)

segment_intervals=(1 10 20 40)

cuda_idx=0

model_path="path_to_model/500000.pt"

for video_mode in "${video_modes[@]}"; do
  for segment_interval in "${segment_intervals[@]}"; do

    echo "-------- Running fs(${segment_interval}) with video_mode: ${video_mode} --------"

    CUDA_VISIBLE_DEVICES=${cuda_idx} xvfb-run -a python3 src/train-ftr.py \
        --log_dir "runs-ablation-segment_interval_${segment_interval}" \
        --algorithm ftr_drq --seed 0 --cuda_idx ${cuda_idx} \
        --domain_name finger --task_name spin \
        --train_steps 200k \
        --supervised_steps 5k --sup_to_rl_warmup_steps 5k \
        --train_mode "video_hard_one_${video_mode}" --eval_mode "video_hard_one_${video_mode}" \
        --save_video --plot_selected --plot_segment \
        --use_selector --use_supervised --use_rl \
        --segment_interval ${segment_interval} \
        --pretrained_model_path "${model_path}"

    echo "-------- Finished fs(${segment_interval}) with video_mode: ${video_mode} --------"
  done
done

echo "All tasks finished."