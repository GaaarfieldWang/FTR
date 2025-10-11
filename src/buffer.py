import numpy as np
import os
import pickle
import random
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from collections import deque
import matplotlib.pyplot as plt

import augmentations


def prefill_memory(obses, capacity, obs_shape, type):
    """Reserves memory for replay buffer"""
    c, h, w = obs_shape
    for i in range(capacity):
        frame = np.ones((c // 3, h, w), dtype=type)
        obses[i] = (frame, frame)
    return obses


class ReplayBuffer(object):
    """Buffer to store environment transitions"""

    def __init__(self, obs_shape, action_shape, capacity, reward_first_capacity, batch_size):
        self.capacity = capacity
        self.reward_first_capacity = reward_first_capacity
        self.batch_size = batch_size

        self._obses = [None] * self.capacity
        self._obses = prefill_memory(
            self._obses, capacity, obs_shape, type=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def _add_observation(self, obs, next_obs, action, reward, done):
        obses = (obs, next_obs)
        self._obses[self.idx] = (obses)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add(self, obses, actions, rewards, next_obses, dones):
        self._add_observation(obses, next_obses, actions, rewards, dones)

    def _get_idxs(self, n=None, reward_first=False, save_steps=1):
        if n is None:
            n = self.batch_size
        if not reward_first:
            return np.random.randint(
                0, self.capacity if self.full else self.idx - save_steps, size=n
            )
        else:
            size = self.capacity if self.full else self.idx - save_steps
            sorted_indexes = sorted(range(size), key=lambda i: self.rewards[i], reverse=True)
            candi_index = sorted_indexes[:min(self.reward_first_capacity, size)]
            return np.random.choice(candi_index, size=n, replace=True)

    def _encode_obses(self, idxs):
        obses, next_obses = [], []
        for i in idxs:
            obs, next_obs = self._obses[i]
            obses.append(np.array(obs, copy=False))
            next_obses.append(np.array(next_obs, copy=False))
        return np.array(obses), np.array(next_obses)

    def __sample__(self, n=None):
        idxs = self._get_idxs(n)

        obs, next_obs = self._encode_obses(idxs)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        return obs, actions, rewards, next_obs, not_dones

    def sample_multi_step(self, step=1, n=None, reward_first=False):
        idxs = self._get_idxs(n, reward_first, step)

        obses, next_obses = [], []
        for i in idxs:
            obs, _ = self._obses[i]
            next_obs, _ = self._obses[(i + step) % self.capacity]
            obses.append(np.array(obs, copy=False))
            next_obses.append(np.array(next_obs, copy=False))
        obs, next_obs = np.array(obses), np.array(next_obses)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        obs = augmentations.random_crop(obs)
        next_obs = augmentations.random_crop(next_obs)

        actions, rewards = [], []
        for add_idx in range(step):
            actions.append(torch.as_tensor(self.actions[(idxs + add_idx) % self.capacity]).cuda())
            rewards.append(torch.as_tensor(self.rewards[(idxs + add_idx) % self.capacity]).cuda())

        return obs, actions, rewards, next_obs

    def sample(self, n=None):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        obs = augmentations.random_crop(obs)
        next_obs = augmentations.random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones

    def __sample_multi_rewards__(self, n=None, n_step=3, def_discount=0.99):
        if n is None:
            n = self.batch_size
        idxs = np.random.randint(0, self.capacity - n_step if self.full else self.idx - n_step, size=n)
        obs, _ = self._encode_obses(idxs)
        _, next_obs = self._encode_obses(idxs + n_step - 1)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        reward = np.zeros_like(self.rewards[idxs])
        discount = np.ones_like(self.not_dones[idxs])
        for i in range(n_step):
            step_reward = self.rewards[idxs + i]
            reward += discount * step_reward
            discount *= self.not_dones[idxs + i] * def_discount
        reward = torch.as_tensor(reward).cuda()
        discount = torch.as_tensor(discount).cuda()
        return obs, actions, reward, discount, next_obs

    def sample_drqv2(self, n=None, n_step=3, discount=0.99):
        obs, action, reward, discount, next_obs = self.__sample_multi_rewards__(n=n, n_step=n_step,
                                                                                def_discount=discount)
        return obs, action, reward, discount, next_obs
    
    def save(self, work_dir):
        with open(os.path.join(work_dir, 'replay_buffer.pkl'), 'wb') as f:
            pickle.dump((self._obses, self.actions, self.rewards, self.not_dones, self.idx, self.full), f)

class SupervisedDataset(Dataset):
    def __init__(self, obses, actions,region_num, channels, h, w):
        self.obses = obses
        self.actions = actions
        self.region_num = region_num
        self.channels = channels
        self.h = h
        self.w = w

    def __len__(self):
        return len(self.obses)

    def __getitem__(self, idx):
        obs = self.obses[idx]
        action = self.actions[idx]
        obs = torch.as_tensor(obs).cuda().float()
        action = torch.as_tensor(action).cuda().long()
        obs, action = self._random_permutation(obs, action)
        return obs, action
    
    def _random_permutation(self, obs, action):
        '''
        obs: torch.Tensor ((region_num + 1) * channels, height, width)
        action: torch.Tensor (region_num,)

        random permutation of regions
        '''
        n = 1
        obs = obs.reshape(n, self.region_num + 1, self.channels, self.h, self.w)
        action = action.reshape(n, self.region_num)
        region_idx = np.random.permutation(self.region_num)
        region_idx_obs = np.concatenate([region_idx, [self.region_num]])
        obs = obs[:, region_idx_obs]
        action = action[:, region_idx]
        return obs.reshape((self.region_num + 1) * self.channels, self.h, self.w), action.reshape(self.region_num,)

class SupervisedBuffer(object):
    """Buffer to store supervised infomaion in high level action"""

    def __init__(self, obs_shape, action_shape, region_num, channels, capacity, batch_size, dir=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.save_interval = 0
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.region_num = region_num
        self.channels = channels
        self.h = obs_shape[-2]
        self.w = obs_shape[-1]
        self.dir = dir

        self.obses = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)

    def _random_permutation(self, obses, actions):
        '''
        obses: torch.Tensor (n, (region_num + 1) * channels, height, width)
        actions: torch.Tensor (n, region_num)

        random permutation of regions
        '''
        n = obses.shape[0]
        obses = obses.reshape(n, self.region_num + 1, self.channels, self.h, self.w)
        actions = actions.reshape(n, self.region_num)
        region_idx = np.random.permutation(self.region_num)
        region_idx_obs = np.concatenate([region_idx, [self.region_num]])
        obses = obses[:, region_idx_obs]
        actions = actions[:, region_idx]
        return obses.reshape(n, (self.region_num + 1) * self.channels, self.h, self.w), actions

    def add(self, obs, action):
        '''
        obs: obs_segments: np.array ((region_num + 1) * channels, height, width)
        action: List[int], len = region_num
        '''
        self.obses.append(obs)
        self.actions.append(np.array(action, dtype=np.int32))

    def sample(self, n=None):
        '''
        return: obses, actions
        obses: torch.Tensor (n, (region_num + 1) * channels, height, width)
        actions: torch.Tensor (n, region_num)
        '''
        if n is None:
            n = self.batch_size
        idxs = np.random.randint(0, len(self.obses), size=n)
        obses = [self.obses[i] for i in idxs]
        actions = [self.actions[i] for i in idxs]
        obses = torch.as_tensor(np.stack(obses)).cuda().float()
        actions = torch.as_tensor(np.stack(actions)).cuda().long()
        obses, actions = self._random_permutation(obses, actions)
        return obses, actions
    
    def data_loader(self):
        '''
        return a DataLoader for supervised learning
        DataLoader: obses, actions
        '''
        dataset = SupervisedDataset(self.obses, self.actions, self.region_num, self.channels, self.h, self.w)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def save(self):
        self.save_interval += 1
        if self.save_interval % 500 == 0:
            self.save_interval = 0
            with open(self.dir + '/supervised_buffer.pkl', 'wb') as f:
                pickle.dump((self.obses, self.actions), f)

class OnPolicyBuffer(object):
    def __init__(self, obs_shape, action_shape, region_num, channels, selector_type):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.region_num = region_num
        self.channels = channels
        self.selector_type = selector_type
        self.h = obs_shape[-2]
        self.w = obs_shape[-1]

        self.obses = deque()
        self.actions = deque()
        self.rewards = deque()
        self.obs_next = deque()
        self.dones = deque()

    def _random_permutation(self, obses, actions):
        '''
        obses: torch.Tensor (n, (region_num + 1) * channels, height, width)
        actions: torch.Tensor (n, region_num)

        random permutation of regions
        '''
        n = obses.shape[0]
        obses = obses.reshape(n, self.region_num + 1, self.channels, self.h, self.w)
        actions = actions.reshape(n, self.region_num)
        region_idx = np.random.permutation(self.region_num)
        region_idx_obs = np.concatenate([region_idx, [self.region_num]])
        obses = obses[:, region_idx_obs]
        actions = actions[:, region_idx]
        return obses.reshape(n, (self.region_num + 1) * self.channels, self.h, self.w), actions

    def add(self, obs, action, reward, obs_next, done):
        self.obses.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.obs_next.append(obs_next)
        self.dones.append(done)

    def sample(self):
        '''
        sample a whole trajectory

        return: obses, actions, rewards, obs_next, dones
        obses: torch.Tensor (n, (region_num + 1) * channels, height, width)
        actions: torch.Tensor (n, region_num)
        rewards: torch.Tensor (n, 1)
        obs_next: torch.Tensor (n, (region_num + 1) * channels, height, width)
        dones: torch.Tensor (n, 1)
        '''
        obses = torch.as_tensor(np.array(self.obses)).cuda().float()
        actions = torch.as_tensor(np.array(self.actions)).cuda()
        rewards = torch.as_tensor(np.array(self.rewards)).view(-1,1).cuda().float()
        obs_next = torch.as_tensor(np.array(self.obs_next)).cuda().float()
        dones = torch.as_tensor(np.array(self.dones)).view(-1,1).cuda().float()
        obses, actions = self._random_permutation(obses, actions)
        if self.selector_type == 2:
            actions = actions.float()
        else:
            actions = actions.long()
        return obses, actions, rewards, obs_next, dones
    
    def clear(self):
        self.obses.clear()
        self.actions.clear()
        self.rewards.clear()
        self.obs_next.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.obses)

class ReplayBufferHighLevel(object):
    def __init__(self, obs_shape, action_shape, region_num, channels, selector_type, capacity, reward_first_capacity, batch_size):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.region_num = region_num
        self.channels = channels
        self.selector_type = selector_type
        assert self.selector_type == 2, 'Only support sac selector'
        self.h = obs_shape[-2]
        self.w = obs_shape[-1]
        self.capacity = capacity
        self.reward_first_capacity = reward_first_capacity
        self.batch_size = batch_size

        self._obses = [None] * self.capacity
        self._obses = prefill_memory(
            self._obses, capacity, ((self.region_num + 1) * self.channels, self.h, self.w), type=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def _random_permutation(self, obses, actions):
        '''
        obses: torch.Tensor (n, (region_num + 1) * channels, height, width)
        actions: torch.Tensor (n, region_num)

        random permutation of regions
        '''
        n = obses.shape[0]
        obses = obses.reshape(n, self.region_num + 1, self.channels, self.h, self.w)
        actions = actions.reshape(n, self.region_num)
        region_idx = np.random.permutation(self.region_num)
        region_idx_obs = np.concatenate([region_idx, [self.region_num]])
        obses = obses[:, region_idx_obs]
        actions = actions[:, region_idx]
        return obses.reshape(n, (self.region_num + 1) * self.channels, self.h, self.w), actions

    def _add_observation(self, obs, next_obs, action, reward, done):
        obses = (obs, next_obs)
        self._obses[self.idx] = (obses)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add(self, obses, actions, rewards, next_obses, dones):
        self._add_observation(obses, next_obses, actions, rewards, dones)

    def _get_idxs(self, n=None, reward_first=False, save_steps=1):
        if n is None:
            n = self.batch_size
        if not reward_first:
            return np.random.randint(
                0, self.capacity if self.full else self.idx - save_steps, size=n
            )
        else:
            size = self.capacity if self.full else self.idx - save_steps
            sorted_indexes = sorted(range(size), key=lambda i: self.rewards[i], reverse=True)
            candi_index = sorted_indexes[:min(self.reward_first_capacity, size)]
            return np.random.choice(candi_index, size=n, replace=True)

    def _encode_obses(self, idxs):
        obses, next_obses = [], []
        for i in idxs:
            obs, next_obs = self._obses[i]
            obses.append(np.array(obs, copy=False))
            next_obses.append(np.array(next_obs, copy=False))
        return np.array(obses), np.array(next_obses)

    def __sample__(self, n=None, reward_first=False):
        idxs = self._get_idxs(n, reward_first=reward_first)

        obs, next_obs = self._encode_obses(idxs)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        return obs, actions, rewards, next_obs, not_dones

    def sample(self, n=None, reward_first=False):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n, reward_first=reward_first)
        obs, actions = self._random_permutation(obs, actions)
        if self.selector_type == 2:
            actions = actions.float()
        else:
            actions = actions.long()

        return obs, actions, rewards, next_obs, not_dones

    def __len__(self):
        return self.capacity if self.full else self.idx
