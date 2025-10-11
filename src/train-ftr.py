import datetime
import gym
import numpy as np
import os
import time
import torch
import warnings

import utils
from algorithms.ftr_drq import FTR_DRQ
from algorithms.factory import make_agent
from arguments import parse_args
from buffer import ReplayBuffer, SupervisedBuffer, OnPolicyBuffer, ReplayBufferHighLevel
from env.wrappers import make_env
from logger import Logger
from video import FTRVideoRecorder as VideoRecorder

import logging

logging.getLogger().setLevel(logging.WARNING)

warnings.filterwarnings("ignore")

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
def reward_scaling(reward):
    return reward

def evaluate(env, agent, video, num_episodes, L, step, mode, domain_name, test_env=False):
    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        obs_selected, obs_segments, high_level_action, last_s, last_a = agent.get_low_level_state(obs, env_reset=True, eval_mode=True)
        video.init(enabled=(i == 0) and step % video.save_freq == 0)
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                if isinstance(agent, FTR_DRQ):
                    action = agent.select_action(obs_selected, step=step, eval_mode=True)
                else:
                    action = agent.select_action(obs_selected)
            obs, reward, done, _ = env.step(action)
            obs_selected, obs_segments, high_level_action, last_s, last_a = agent.get_low_level_state(obs, eval_mode=True)
            if video.plot_segment or video.plot_selected:
                logits, selected_image = agent.select_image(
                    obs_segments)
                selected_image = agent.select_image_for_plot(obs_selected)
            if video.plot_segment:
                segmented_image = utils.transfer_to_ann(obs_segments, logits, video.channels,
                                                        video.region_num,
                                                        video.stack_num)
            else:
                segmented_image = None
            if not video.plot_selected:
                selected_image = None
            high_level_selection = utils.transfer_to_ann_high_level(last_s, last_a, video.channels, video.region_num)
            if "robosuite" in domain_name:
                video.record(env, mode=mode, selected_image=selected_image, segmented_image=segmented_image, original_image=obs_segments, high_level_selection=high_level_selection)
            else:
                video.record(env, mode=mode, selected_image=selected_image, segmented_image=segmented_image, high_level_selection=high_level_selection)
            episode_reward += reward

        if L is not None:
            _test_env = '_test_env' if test_env else ''
            video.save(f'{step}{_test_env}.mp4')
        episode_rewards.append(episode_reward)
    if L is not None:
        _test_env = '_test_env' if test_env else ''
        L.log(f'eval/episode_reward{_test_env}', np.mean(episode_rewards), step)
    return np.mean(episode_rewards)


def main(args):
    # Set seed
    utils.set_seed_everywhere(args.seed)

    # Initialize environments
    gym.logger.set_level(40)
    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode=args.train_mode,
        color_type=args.color_type,
        apply_sam=args.apply_sam,
        args=args,
        is_train_env=True
    )
    test_env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed + 42,
        episode_length=args.episode_length,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode=args.eval_mode,
        color_type=args.color_type,
        apply_sam=args.apply_sam,
        args=args,
        is_train_env=False
    ) if args.eval_mode is not None else env

    # Create working directory
    work_dir = os.path.join(args.log_dir, args.domain_name + '_' + args.task_name, args.algorithm, curr_time)
    print('Working directory:', work_dir)
    assert not os.path.exists(os.path.join(
        work_dir, 'train.log')), 'specified working directory already exists'
    utils.make_dir(work_dir)
    if not os.path.exists("figures"):
        utils.make_dir("figures")
    model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
    video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
    video = VideoRecorder(video_dir if args.save_video else None, args.plot_segment, args.plot_selected, height=84,
                          width=84, channels=args.channels, region_num=args.masked_region_num,
                          stack_num=args.frame_stack,
                          save_freq=args.save_video_freq)
    utils.write_info(args, os.path.join(work_dir, 'info.log'))

    # Prepare agent
    assert torch.cuda.is_available(), 'must have cuda enabled'
    obs_shape = env.observation_space.shape # (S * C, H, W)
    true_obs_shape = list(obs_shape)
    true_obs_shape[0] = true_obs_shape[0] * args.masked_region_num + args.channels * args.frame_stack if args.add_original_frame else true_obs_shape[0]
    # true_obs_shape = (S * (R + 1) * C, H, W)
    agent_obs_shape = list(obs_shape)
    agent_obs_shape[0] = agent_obs_shape[0] * args.masked_region_num
    # agent_obs_shape = (S * R * C, H, W)

    high_level_obs_shape = list(obs_shape)
    high_level_obs_shape[0] = args.channels * (args.masked_region_num + 1) if args.add_original_frame else args.channels * args.masked_region_num

    low_level_obs_shape = list(obs_shape)
    # low_level_obs_shape = (S * C, H, W)

    action_shape = env.action_space[0].shape if isinstance(env.action_space,
                                                           gym.spaces.tuple.Tuple) else env.action_space.shape
    assert len(obs_shape) >= 3, "Dimension of observation must be 3 or 4"
    if len(obs_shape) == 4:
        obs_shape = obs_shape[1:]
    assert len(action_shape) <= 2, "Dimension of action must be 1 or 2"
    if len(action_shape) == 2:
        action_shape = action_shape[1:]
    print('Observation space:', obs_shape)
    print('Agent obs shape:', agent_obs_shape)
    print('Buffer obs shape: ', true_obs_shape)
    print('High level obs shape:', high_level_obs_shape)
    print('Low level obs shape:', low_level_obs_shape)
    print('Action space:', action_shape)
    agent = make_agent(
        obs_shape=low_level_obs_shape,
        action_shape=action_shape,
        args=args
    )
    assert isinstance(agent, FTR_DRQ), \
        'agent must be FTR_DRQ'
    if args.pretrained_model_path is not None:
        static_dict = torch.load(args.pretrained_model_path)
        agent.load_state_dict(static_dict)
        print('load static dict')
    replay_buffer = ReplayBuffer(
        obs_shape=low_level_obs_shape,
        action_shape=action_shape,
        capacity=args.capacity,
        reward_first_capacity=args.reward_first_capacity,
        batch_size=args.batch_size
    )

    supervised_buffer = SupervisedBuffer(
        obs_shape=high_level_obs_shape,
        action_shape=[args.masked_region_num],
        region_num=args.masked_region_num,
        channels=args.channels,
        capacity=args.capacity,
        batch_size=args.batch_size,
        dir=os.path.join(args.log_dir, args.domain_name + '_' + args.task_name, args.algorithm)
    )
    if args.sac_selector:
        high_level_buffer = ReplayBufferHighLevel(
            obs_shape=high_level_obs_shape,
            action_shape=(args.masked_region_num, ),
            region_num=args.masked_region_num,
            channels=args.channels,
            selector_type=args.selector_type,
            capacity=args.capacity,
            reward_first_capacity=args.reward_first_capacity,
            batch_size=args.batch_size
    )
    else:
        high_level_buffer = OnPolicyBuffer(
            obs_shape=high_level_obs_shape,
            action_shape=[args.masked_region_num],
            region_num=args.masked_region_num,
            channels=args.channels,
            selector_type=args.selector_type,
        )

    print("=====Start training=====")
    done = False
    start_time = time.time()
    episode = 0
    episode_reward = 0
    episode_step = 0
    step = 0
    obs = env.reset()
    obs_selected, obs_segments, high_level_action, _, _ = agent.get_low_level_state(obs, env_reset=True, step=step)
    h_reward = 0
    h_state = obs_segments[:args.channels * (args.masked_region_num + 1)]
    h_action = high_level_action
    L = Logger(work_dir, args)
    unsaved_supervised = True
    print('Evaluating:', work_dir)
    if test_env is not None:
        evaluate(test_env, agent, video, args.eval_episodes, L, step,
                    args.eval_mode, args.domain_name, test_env=True)
    L.dump(step)
    while step < args.train_steps:
        # add supervised information for high level agent
        if args.use_supervised and step < args.supervised_steps:
            ground_truth = agent.get_ground_truth_high_level_action(obs_segments[:args.channels * (args.masked_region_num + 1)])
            supervised_buffer.add(obs_segments[:args.channels * (args.masked_region_num + 1)], ground_truth)
        if unsaved_supervised and step > args.supervised_steps:
            agent.save_supervised_info(work_dir)
            unsaved_supervised = False
        while not done and not agent.time_to_segment():
            '''
            either done or time to segment will exit the loop
            '''
            if step < args.init_steps:
                action = env.action_space.sample()
            else:
                with utils.eval_mode(agent):
                    if isinstance(agent, FTR_DRQ):
                        action = agent.select_action(obs_selected, step=step, eval_mode=False)
                    else:
                        action = agent.sample_action(obs_selected)
                    action = action.flatten()

            if args.train_agent and step > args.init_steps:
                agent.update(replay_buffer, L, step)

            next_obs, reward, done, _ = env.step(action)
            next_obs_selected, next_obs_segments, high_level_action, _, _ = agent.get_low_level_state(next_obs, step=step)
            done_bool = 0 if episode_step + 1 == args.max_episode_steps else float(done)
            episode_reward = episode_reward + reward
            h_reward = h_reward + reward
            if args.train_agent:
                replay_buffer.add(obs_selected, action, reward, next_obs_selected, done_bool)
            episode_step = episode_step + 1
            step = step + 1
            obs = next_obs
            obs_selected = next_obs_selected
            obs_segments = next_obs_segments

        if done:
            if step > 0:
                L.log('train/duration', time.time() -
                      start_time, step)
                L.dump(step)

            # Evaluate agent periodically
            if step % args.eval_freq == 0:
                print('Evaluating:', work_dir)
                if test_env is not None:
                    evaluate(test_env, agent, video, args.eval_episodes, L, step,
                             args.eval_mode, args.domain_name, test_env=True)
                L.dump(step)

            # Save agent periodically
            if step > 0 and step % args.save_freq == 0:
                torch.save(agent.state_dict(), os.path.join(model_dir, f'{step}.pt'))

            L.log('train/episode_reward', episode_reward, step)

            start_time = time.time()
            episode_reward = 0
            episode_step = 0
            episode = episode + 1
            if args.use_selector:
                high_level_buffer.add(h_state, h_action, reward_scaling(h_reward), obs_segments[:args.channels * (args.masked_region_num + 1)], done)
                L.log('train/h_reward', h_reward, step)
                agent.update_high_level(high_level_buffer, supervised_buffer, L, step)
            obs = env.reset()
            done = False
            obs_selected, obs_segments, high_level_action, _, _ = agent.get_low_level_state(obs, env_reset=True, step=step)
            h_state = obs_segments[:args.channels * (args.masked_region_num + 1)]
            h_action = high_level_action
            h_reward = 0
            
        elif agent.time_to_segment():
            '''
            next call of agent.get_low_level_state() will use segment selector to select the next segment
            '''
            if step < args.init_steps:
                action = env.action_space.sample()
            else:
                with utils.eval_mode(agent):
                    if isinstance(agent, FTR_DRQ):
                        action = agent.select_action(obs_selected, step=step, eval_mode=False)
                    else:
                        action = agent.sample_action(obs_selected)
                    action = action.flatten()

            if args.train_agent and step > args.init_steps:
                agent.update(replay_buffer, L, step)

            next_obs, reward, done, _ = env.step(action)
            next_obs_selected, next_obs_segments, high_level_action, _, _ = agent.get_low_level_state(next_obs, step=step)
            done_bool = 0 if episode_step + 1 == args.max_episode_steps else float(done)
            episode_reward = episode_reward + reward
            h_reward = h_reward + reward
            if args.train_agent:
                replay_buffer.add(obs_selected, action, reward, next_obs_selected, done_bool)
            next_h_state = next_obs_segments[:args.channels * (args.masked_region_num + 1)]
            if args.use_selector:
                high_level_buffer.add(h_state, h_action, reward_scaling(h_reward), next_h_state, done)
                L.log('train/h_reward', h_reward, step)
                agent.update_high_level(high_level_buffer, supervised_buffer, L, step)
            episode_step = episode_step + 1
            step = step + 1
            obs = next_obs
            obs_selected = next_obs_selected
            obs_segments = next_obs_segments
            h_state = next_h_state
            h_action = high_level_action
            h_reward = 0

    torch.save(agent.state_dict(), os.path.join(model_dir, f'{args.train_steps}.pt'))

    print('Completed training for', work_dir)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    args.description = "_".join(
        [args.algorithm, args.train_mode, str(args.seed)])
    main(args)
