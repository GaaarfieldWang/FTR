import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
import matplotlib.pyplot as plt
import algorithms.modules as m
import net.auxiliary_pred as aux
import utils

class HighLevelSAC(object):
    '''
    use sac to train the high level segment selector
    '''
    def __init__(self,args, segment_selector, critic):
        self.selector_type = args.selector_type
        assert self.selector_type == 2, 'SAC only supports continuous selector'
        self.use_supervised = args.use_supervised
        self.batch_size = args.sac_selector_batch_size
        self.lr = args.sac_selector_lr
        self.init_steps = args.sac_init_steps
        self.supervised_steps = args.supervised_steps
        self.gamma = args.gamma
        self.critic_tau = args.sac_selector_critic_tau
        self.encoder_tau = args.sac_selector_encoder_tau
        self.actor_update_freq = args.sac_selector_update_freq
        self.critic_target_update_freq = args.sac_selector_target_update_freq
        self.max_grad_norm = args.max_grad_norm

        self.reward_scaling = args.ppo_reward_scaling
        self.reward_scale = args.segment_interval

        self.reward_first = args.sac_selector_reward_first_sampling

        self.actor = segment_selector
        self.critic = critic

        self.critic_target = deepcopy(self.critic)

        self.log_alpha = torch.tensor(np.log(args.init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(args.masked_region_num)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.sac_selector_lr, betas=(args.actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.sac_selector_critic_lr, betas=(args.critic_beta, 0.999)
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=args.sac_selector_alpha_lr, betas=(args.alpha_beta, 0.999)
        )
        self.supervised_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=args.sac_selector_lr,
            betas=(args.selector_beta, 0.999)
        )

        self.train()
        self.critic_target.train()
    
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
    
    def eval(self):
        self.train(False)

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def _sample_action_from_probs(self, raw_probs, eval_mode=False, step=None):
        '''
        raw_probs: torch.Tensor, shape = (B, 1, region_num, 2)
        The last dimension is the probability of taking actions 0 and 1, meaning selecting or not selecting the region

        return: List[bool], shape = (B, 1, region_num)
        '''
        return self.actor._sample_action_from_probs(raw_probs, eval_mode, step)
    
    def compute_log_probs(self, raw_probs, action, std=0.05):
        '''
        raw_probs: torch.Tensor, shape = (B, 1, region_num, 2)

        action: torch.Tensor, shape = (B, region_num)

        return:
        log_probs: torch.Tensor, shape = (B, )
        entropy: torch.Tensor, shape = (B, )
        '''
        if self.selector_type == 0:
            return self.compute_log_probs_softmax(raw_probs, action)
        elif self.selector_type == 2:
            return self.compute_log_probs_continuous(raw_probs, action, std)
        assert raw_probs.shape[2] == action.shape[1]
        B = raw_probs.shape[0]
        raw_probs = raw_probs.squeeze(1)    # shape = (B, region_num, 2)
        region_num = raw_probs.shape[1]
        multi_categoricals = [torch.distributions.Categorical(probs=raw_probs[:, i, :]) for i in range(region_num)] # shape = (region_num, B)
        log_probs = torch.stack([multi_categoricals[i].log_prob(action[:, i]) for i in range(region_num)], dim=1)   # shape = (B, region_num)
        entropy = torch.stack([multi_categoricals[i].entropy() for i in range(region_num)], dim=1)  # shape = (B, region_num)
        log_probs = log_probs.sum(1)    # shape = (B, )
        entropy = entropy.sum(1)    # shape = (B, )
        return log_probs.view(-1, 1), entropy.view(-1, 1)
    
    def compute_log_probs_softmax(self, raw_probs, action):
        '''
        raw_probs: torch.Tensor, shape = (B, 1, region_num, 2)

        action: torch.Tensor, shape = (B, region_num)

        return:
        log_probs: torch.Tensor, shape = (B, )
        entropy: torch.Tensor, shape = (B, )
        '''
        assert raw_probs.shape[2] == action.shape[1]
        B = raw_probs.shape[0]
        for act in action:
            assert torch.sum(act) == 1
        selected = torch.argmax(action, dim=1)

        probs = raw_probs[:, :, :, 1]   # shape = (B, 1, region_num)
        probs = probs.squeeze(1)    # shape = (B, region_num)
        m = torch.distributions.Categorical(probs=probs)
        log_probs = m.log_prob(selected)    # shape = (B, )
        entropy = m.entropy()   # shape = (B, )
        return log_probs.view(-1, 1), entropy.view(-1, 1)
    
    def compute_log_probs_continuous(self, raw_probs, action, std=0.05):
        '''
        raw_probs: torch.Tensor, shape = (B, 1, region_num, 2)

        action: torch.Tensor, shape = (B, region_num)

        return:
        log_probs: torch.Tensor, shape = (B, )
        entropy: torch.Tensor, shape = (B, )
        '''
        assert raw_probs.shape[2] == action.shape[1]
        B = raw_probs.shape[0]
        probs = raw_probs[:, :, :, 1]   # shape = (B, 1, region_num)
        probs = probs.squeeze(1)    # shape = (B, region_num)
        m = torch.distributions.Normal(probs, std)
        log_probs = m.log_prob(action)    # shape = (B, region_num)
        entropy = m.entropy()   # shape = (B, region_num)
        log_probs = log_probs.sum(1)    # shape = (B, )
        entropy = entropy.sum(1)    # shape = (B, )
        return log_probs.view(-1, 1), entropy.view(-1, 1)
    
    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        """
        The soft value function V is defined as:
        V(s) = Expected value over actions from policy of [Q(s, a) - alpha * log(policy(a|s))]
        """
        with torch.no_grad():
            # _, policy_action, log_pi, _ = self.actor(next_obs)
            _, next_raw_probs, next_std = self.actor(next_obs, return_raw_probs=True, step=step)
            # continuous action space
            next_action = self._sample_action_from_probs(next_raw_probs, eval_mode=False, step=step)
            next_action = torch.tensor(next_action).float().cuda()  # (B, 1, region_num)
            next_action = next_action.squeeze(1)    # (B, region_num)
            policy_action = next_action
            log_pi, _ = self.compute_log_probs(next_raw_probs, policy_action, next_std)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.gamma * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        if L is not None:
            L.log('train_selector/SAC_critic_loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
        # _, pi, log_pi, log_std = self.actor(obs, detach=True)
        _, raw_probs, std = self.actor(obs, return_raw_probs=True, detach=True, step=step)
        pi = self._sample_action_from_probs(raw_probs, eval_mode=False, step=step)
        pi = torch.tensor(pi).float().cuda()
        pi = pi.squeeze(1)  # (B, region_num)
        log_pi, _ = self.compute_log_probs(raw_probs, pi, std)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if L is not None:
            L.log('train_selector/SAC_actor_loss', actor_loss, step)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (
                    self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L.log('train_selector/SAC_alpha_loss', alpha_loss, step)
                L.log('train_selector/SAC_alpha_value', self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def soft_update_critic_target(self):
        utils.soft_update_params(
            self.critic.Q1, self.critic_target.Q1, self.critic_tau
        )
        utils.soft_update_params(
            self.critic.Q2, self.critic_target.Q2, self.critic_tau
        )
        utils.soft_update_params(
            self.critic.region_embedding, self.critic_target.region_embedding,
            self.encoder_tau
        )

    def update_supervised(self, supervised_buffer, L, step):
        losses = []
        coef = 1.0
        if step < self.supervised_steps:
            coef = 1.0
        elif step < 2 * self.supervised_steps:
            coef = 2.0 - step / self.supervised_steps
        else:
            coef = 0.0
        if coef < 1e-3:
            return
        
        # lr decay
        self.supervised_optimizer.param_groups[0]['lr'] = self.lr * coef

        data_loader = supervised_buffer.data_loader()
        for _ in range(1):
            for obs_segments, ground_truth_high_level_action in data_loader:
                _, raw_probs = self.actor(obs_segments, return_raw_probs=True)
                # raw_probs: torch.Tensor, shape = (B, 1, region_num, 2)
                # ground_truth_high_level_action: torch.Tensor (B, region_num)
                
                # use bce loss
                raw_probs = raw_probs.reshape(-1, 2)[:, 1]
                ground_truth_high_level_action = ground_truth_high_level_action.reshape(-1)
                loss = F.binary_cross_entropy(raw_probs, ground_truth_high_level_action.float())
                losses.append(loss)
                
                self.supervised_optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.supervised_optimizer.step()
        L.log('train/supervised_loss', torch.stack(losses).mean(), step)

    def update(self, buffer, supervised_buffer, L, step):
        if step < self.init_steps:
            return
        obs, action, reward, next_obs, not_done = buffer.sample(n=self.batch_size, reward_first=self.reward_first)
        if self.reward_scaling:
            reward = reward / self.reward_scale
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        self.update_actor_and_alpha(obs, L, step)
        self.soft_update_critic_target()
        if self.use_supervised:
            self.update_supervised(supervised_buffer, L, step)
        L.log('train_selector/selector_std', self.actor.std(step), step)

    def state_dict(self):
        return {
            # 'segment_selector': self.actor.state_dict(),
            # 'critic': self.critic.state_dict(),
            # 'critic_target': self.critic_target.state_dict(),
            # 'log_alpha': self.log_alpha,
            # actor and critic are already loaded
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha_optimizer': self.log_alpha_optimizer.state_dict(),
            'supervised_optimizer': self.supervised_optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['segment_selector'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        self.log_alpha = state_dict['log_alpha']
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        self.log_alpha_optimizer.load_state_dict(state_dict['log_alpha_optimizer'])
        self.supervised_optimizer.load_state_dict(state_dict['supervised_optimizer'])
