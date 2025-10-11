#!/bin/bash

# Run finger spin with different supervised_steps. Our default is 5k.

video_modes=(0 1 2 3 4)

supervised_stepss=(1000 5000 10000)

cuda_idx=0

model_path="path_to_model/500000.pt"

for video_mode in "${video_modes[@]}"; do
  for supervised_steps in "${supervised_stepss[@]}"; do

    echo "-------- Running fs(${supervised_steps}) with video_mode: ${video_mode} --------"

    CUDA_VISIBLE_DEVICES=${cuda_idx} xvfb-run -a python3 src/train-ftr.py \
        --log_dir "runs-ablation-vlm_steps_${supervised_steps}" \
        --algorithm ftr_drq --seed 0 --cuda_idx ${cuda_idx} \
        --domain_name finger --task_name spin \
        --train_steps 200k \
        --supervised_steps ${supervised_steps} --sup_to_rl_warmup_steps $((10001 - supervised_steps)) \
        --train_mode "video_hard_one_${video_mode}" --eval_mode "video_hard_one_${video_mode}" \
        --save_video --plot_selected --plot_segment \
        --use_selector --use_supervised --use_rl \
        --segment_interval 20 \
        --pretrained_model_path "${model_path}"

    echo "-------- Finished fs(${supervised_steps}) with video_mode: ${video_mode} --------"
  done
done

echo "All tasks finished."