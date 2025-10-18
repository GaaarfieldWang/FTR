# Focus-Then-Reuse: Fast Adaptation in Visual Perturbation Environments

This repository contains the code for the NeurIPS 2025 paper: Focus-Then-Reuse: Fast Adaptation in Visual Perturbation Environments.

## Installation

You can directly install the dependencies using the provided `setup.sh` file.

```sh
./setup.sh
```

Please refer to [this page](https://help.aliyun.com/zh/model-studio/get-api-key) to acquire the API key, and then set the environment variable `DASHSCOPE_API_KEY` to your API key.

```sh
export DASHSCOPE_API_KEY="your-api-key"
```

Please run the following command to download the pretrained SAM 2 model weights:

```sh
./src/segment-anything-2-real-time/checkpoints/download_ckpts.sh
```

More details can be found in the official [SAM-2](https://github.com/facebookresearch/sam2?tab=readme-ov-file#sam-2-checkpoints) repository.

## Run

Below are running commands of running FTR and baseline algorithms. 

1. Train the original policy in clean environment using DrQ-v2 algorithm.

    Please replace *env* and *task* with corresponding arguments 
(e.g. ./scripts/train.sh franka reach). 

    ```sh
    ./scripts/train.sh env task
    ```

    The model will be saved in `./log_dir/env_task/algorithm/time/model` folder.

2. Adapt the original policy in the perturbed environment using FTR.

    Please first refer to [this page](https://help.aliyun.com/zh/model-studio/get-api-key) to acquire the API key, and then `export DASHSCOPE_API_KEY="your-api-key"`.

    Then fill in the `adapt-all.sh` file with the corresponding model paths.

    ```sh
    ./scripts/adapt-all.sh
    ```


3. To run the ablation study, please refer to `./scripts/ablation-*.sh`

For more parameter settings, please refer to `arguments.py`.

If you encounter problems related to rendering, please refer to respiratory of 
[DMC](https://github.com/google-deepmind/dm_control)

## Visualization

<div style="display: flex; justify-content: center;">
    <div style="flex: 0 0 100%; text-align: center;">
        <figure>
            <img src="./assets/ps.gif" alt="pendulum_swingup" style="width: 100%;">
            <figcaption>pendulum-swingup</figcaption>
        </figure>
    </div>
</div>

<div style="display: flex; justify-content: center;">
    <div style="flex: 0 0 100%; text-align: center;">
        <figure>
            <img src="./assets/cs.gif" alt="cartpole_swingup" style="width: 100%;">
            <figcaption>cartpole-swingup</figcaption>
        </figure>
    </div>
</div>

<div style="display: flex; justify-content: center;">
    <div style="flex: 0 0 100%; text-align: center;">
        <figure>
            <img src="./assets/fs.gif" alt="finger_spin" style="width: 100%;">
            <figcaption>finger-spin</figcaption>
        </figure>
    </div>
</div>

<div style="display: flex; justify-content: center;">
    <div style="flex: 0 0 100%; text-align: center;">
        <figure>
            <img src="./assets/hs.gif" alt="hopper_stand" style="width: 100%;">
            <figcaption>hopper-stand</figcaption>
        </figure>
    </div>
</div>

<div style="display: flex; justify-content: center;">
    <div style="flex: 0 0 100%; text-align: center;">
        <figure>
            <img src="./assets/hh.gif" alt="hopper_hop" style="width: 100%;">
            <figcaption>hopper-hop</figcaption>
        </figure>
    </div>
</div>

<div style="display: flex; justify-content: center;">
    <div style="flex: 0 0 100%; text-align: center;">
        <figure>
            <img src="./assets/cr.gif" alt="cheetah_run" style="width: 100%;">
            <figcaption>cheetah-run</figcaption>
        </figure>
    </div>
</div>

<div style="display: flex; justify-content: center;">
    <div style="flex: 0 0 100%; text-align: center;">
        <figure>
            <img src="./assets/ww.gif" alt="walker_walk" style="width: 100%;">
            <figcaption>walker-walk</figcaption>
        </figure>
    </div>
</div>

<div style="display: flex; justify-content: center;">
    <div style="flex: 0 0 100%; text-align: center;">
        <figure>
            <img src="./assets/wr.gif" alt="walker_run" style="width: 100%;">
            <figcaption>walker-run</figcaption>
        </figure>
    </div>
</div>

<div style="display: flex; justify-content: center;">
    <div style="flex: 0 0 100%; text-align: center;">
        <figure>
            <img src="./assets/reach.gif" alt="franka_reach" style="width: 100%;">
            <figcaption>franka-reach</figcaption>
        </figure>
    </div>
</div>

<div style="display: flex; justify-content: center;">
    <div style="flex: 0 0 100%; text-align: center;">
        <figure>
            <img src="./assets/push.gif" alt="franka_push" style="width: 100%;">
            <figcaption>franka-push</figcaption>
        </figure>
    </div>
</div>

<div style="display: flex; justify-content: center;">
    <div style="flex: 0 0 100%; text-align: center;">
        <figure>
            <img src="./assets/door.gif" alt="franka_door" style="width: 100%;">
            <figcaption>franka-door</figcaption>
        </figure>
    </div>
</div>


## Acknowledgements
This code is built on top of the repositorys: [DMControl Generalization Benchmark](https://github.com/nicklashansen/dmcontrol-generalization-benchmark) and [FTD](https://github.com/LAMDA-RL/FTD).

The PPO implementation is based on [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

The DrQ-v2 implementation is based on [DrQ-v2](https://github.com/facebookresearch/drqv2).

The SAM implementation is based on [SAM-2 real-time](https://github.com/Gy920/segment-anything-2-real-time).
