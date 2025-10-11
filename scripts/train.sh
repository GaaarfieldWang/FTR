CUDA_VISIBLE_DEVICES=0 xvfb-run -a python3 src/train-ftr.py \
    --log_dir runs-clean \
    --algorithm ftr_drq --seed 0 --cuda_idx 0 \
    --train_steps 500k \
    --domain_name $1 --task_name $2 \
    --train_mode video_black --eval_mode video_black \
    --save_video --plot_selected --plot_segment \
    --use_rl \
    --segment_interval 20 \
    --train_agent