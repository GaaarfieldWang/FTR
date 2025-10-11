import numpy as np
import torch
import torch.nn.functional as F
import algorithms.modules as m

class HighLevelPPO(object):
    '''
    use ppo to train the high level segment selector
    '''
    def __init__(self,args, segment_selector, critic):
        self.selector_type = args.selector_type
        self.use_supervised = args.use_supervised
        self.use_rl = args.use_rl
        self.ppo_epoch = args.ppo_epoch
        self.batch_size = args.ppo_batch_size
        self.lr = args.ppo_lr
        self.mini_batch_size = args.ppo_mini_batch_size
        self.supervised_steps = args.supervised_steps
        self.warmup_steps = args.sup_to_rl_warmup_steps
        self.bce_loss = args.bce_loss
        assert self.batch_size % self.mini_batch_size == 0, 'batch_size should be divisible by mini_batch_size'
        self.gamma = args.gamma
        self.max_grad_norm = args.max_grad_norm
        self.eps = args.clip_ratio
        self.critic_coef = args.critic_coef
        self.ent_coef = args.ent_coef
        self.reward_scaling = args.ppo_reward_scaling
        self.clip_vloss = args.ppo_clip_vloss
        self.target_kl = args.target_kl
        self.reward_scale = args.segment_interval
        # for GAE
        self.lmbda = args.lmbda
        self.norm_adv = args.norm_adv

        self.segment_selector = segment_selector
        self.critic = critic
        self.actor_critic = m.ActorCritic(self.segment_selector, self.critic)
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), 
            lr=args.ppo_lr, 
            betas=(args.selector_beta, 0.999)
        )
        self.supervised_optimizer = torch.optim.Adam(
            self.segment_selector.parameters(),
            lr=args.ppo_lr,
            betas=(args.selector_beta, 0.999)
        )
    
    def train(self, training=True):
        self.training = training
        self.segment_selector.train(training)
        self.critic.train(training)
    
    def eval(self):
        self.train(False)

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
    
    def ref_compute_log_probs(self, raw_probs, action):
        '''
        raw_probs: torch.Tensor, shape = (B, 1, region_num, 2)

        action: torch.Tensor, shape = (B, region_num)

        return:
        log_probs: torch.Tensor, shape = (B, )
        entropy: torch.Tensor, shape = (B, )
        '''
        assert raw_probs.shape[2] == action.shape[1]
        B = raw_probs.shape[0]
        raw_probs = raw_probs.reshape(-1, 2)
        raw_probs = torch.clamp(raw_probs, min=1e-8)
        log_probs = torch.log(raw_probs.gather(1, action.view(-1, 1)).squeeze(1))   # shape = (B*region_num, )
        log_probs = log_probs.reshape(B, -1).sum(1)  # shape = (B, ), sum over region_num
        log_probs = log_probs.view(-1, 1)
        entropy = -(raw_probs * torch.log(raw_probs)).sum(1)    # shape = (B*region_num, )
        entropy = entropy.reshape(B, -1).sum(1) # shape = (B, ), sum over region_num
        entropy = entropy.view(-1, 1)
        return log_probs, entropy
    
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

    def update_supervised(self, supervised_buffer, L, step):
        losses = []
        coef = 1.0
        if step < self.supervised_steps:
            coef = 1.0
        elif step < self.supervised_steps + self.warmup_steps:
            coef = 1.0 - (step - self.supervised_steps) / self.warmup_steps
        else:
            coef = 0.0
        if coef < 1e-3:
            return
        
        # lr decay
        self.supervised_optimizer.param_groups[0]['lr'] = self.lr * coef

        data_loader = supervised_buffer.data_loader()
        for _ in range(1):
            for obs_segments, ground_truth_high_level_action in data_loader:
                _, raw_probs = self.segment_selector(obs_segments, return_raw_probs=True)
                # raw_probs: torch.Tensor, shape = (B, 1, region_num, 2)
                # ground_truth_high_level_action: torch.Tensor (B, region_num)
                raw_probs = raw_probs.reshape(-1, 2)[:, 1]  # shape = (B*region_num, )
                ground_truth_high_level_action = ground_truth_high_level_action.reshape(-1) # shape = (B*region_num, )
                
                if self.bce_loss:
                    # use bce loss
                    loss = F.binary_cross_entropy(raw_probs, ground_truth_high_level_action.float())
                else:
                    # use clip loss
                    # loss = torch.mean(
                    #     ground_truth_high_level_action * torch.relu(0.6 - raw_probs) +
                    #     (1 - ground_truth_high_level_action) * torch.relu(raw_probs - 0.4)
                    # )
                    loss = m.clip_loss(raw_probs, ground_truth_high_level_action)
                
                losses.append(loss)
                
                self.supervised_optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.segment_selector.parameters(), self.max_grad_norm)
                self.supervised_optimizer.step()
        L.log('train/supervised_loss', torch.stack(losses).mean(), step)

    def update(self, buffer, supervised_buffer, L, step):
        if self.use_supervised and step < self.supervised_steps:
            self.update_supervised(supervised_buffer, L, step)
            buffer.clear()
            return
        if len(buffer) < self.batch_size:
            return
        assert len(buffer) == self.batch_size
        critic_coef = self.critic_coef
        if self.use_rl:
            if self.use_supervised:
                coef = 1.0
                if step > self.supervised_steps and step < self.supervised_steps + self.warmup_steps:
                    # learning rate warmup linearly: 0x -> 1x
                    coef = (step - self.supervised_steps) / self.warmup_steps
                    # critic_coef warmup linearly: 20x -> 1x
                    critic_coef = 20 - (step - self.supervised_steps) * 19 / self.warmup_steps
                    critic_coef = critic_coef * self.critic_coef
                self.optimizer.param_groups[0]['lr'] = self.lr * coef
            state, action, reward, next_state, done = buffer.sample()
            # reward scaling
            if self.reward_scaling:
                reward = reward / self.reward_scale
            with torch.no_grad():
                # compute advantage, using GAE
                values = self.critic(state)
                advantage = torch.zeros_like(reward).to(reward.device)
                last_gae_lam = 0
                for t in reversed(range(self.batch_size)):
                    if t == self.batch_size - 1:
                        next_not_terminal = 1.0 - done[t]
                        next_value = self.critic(next_state[-1:])[0]
                    else:
                        next_not_terminal = 1.0 - done[t]
                        next_value = values[t+1]
                    if torch.abs(next_not_terminal - 0.0) < 1e-3:
                        next_value = 0
                    delta = reward[t] + self.gamma * next_value * next_not_terminal - values[t]
                    advantage[t] = last_gae_lam = delta + self.gamma * self.lmbda * next_not_terminal * last_gae_lam
                td_target = advantage + values
                # compute old_log_probs
                _, old_raw_probs, old_std = self.segment_selector(state, return_raw_probs=True, step=step)
                old_log_probs, _ = self.compute_log_probs(old_raw_probs, action, old_std)
                values = values.detach()
                advantage = advantage.detach()
                td_target = td_target.detach()
                old_log_probs = old_log_probs.detach()
            actor_losses, critic_losses, ent_losses, losses = [], [], [], []
            idxs = np.arange(self.batch_size)
            clip_fracs = []
            for _ in range(self.ppo_epoch):
                np.random.shuffle(idxs)
                for start in range(0, self.batch_size, self.mini_batch_size):
                    mini_batch_idx = idxs[start:start+self.mini_batch_size]
                    _, mb_raw_probs, mb_std = self.segment_selector(state[mini_batch_idx], return_raw_probs=True, step=step)
                    mb_log_probs, mb_entropy = self.compute_log_probs(mb_raw_probs, action[mini_batch_idx], mb_std)
                    mb_log_ratio = mb_log_probs - old_log_probs[mini_batch_idx]
                    mb_ratio = mb_log_ratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-mb_log_ratio).mean()
                        approx_kl = ((mb_ratio - 1) - mb_log_ratio).mean()
                        clip_fracs += [((mb_ratio - 1.0).abs() > self.eps).float().mean().item()]

                    mb_advantage = advantage[mini_batch_idx]
                    if self.norm_adv:
                        mb_advantage = (mb_advantage - mb_advantage.mean()) / (mb_advantage.std() + 1e-8)
                    surr1 = mb_ratio * mb_advantage
                    surr2 = torch.clamp(mb_ratio, 1-self.eps, 1+self.eps) * mb_advantage
                    actor_loss = -torch.min(surr1, surr2).mean()
                    mb_new_values = self.critic(state[mini_batch_idx])
                    if self.clip_vloss:
                        critic_loss_unclipped = (mb_new_values - td_target[mini_batch_idx]) ** 2
                        v_clipped = values[mini_batch_idx] + torch.clamp(mb_new_values - values[mini_batch_idx], -self.eps, self.eps)
                        critic_loss_clipped = (v_clipped - td_target[mini_batch_idx]) ** 2
                        critic_loss = 0.5 * torch.max(critic_loss_unclipped, critic_loss_clipped).mean()
                    else:
                        critic_loss = F.mse_loss(mb_new_values, td_target[mini_batch_idx])
                    entropy_loss = -mb_entropy.mean()
                    loss = actor_loss + critic_coef * critic_loss + self.ent_coef * entropy_loss
                    if torch.isnan(loss):
                        import pdb; pdb.set_trace()
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    ent_losses.append(entropy_loss)
                    losses.append(loss)
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.max_grad_norm != 0:
                        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                if self.target_kl > 0 and approx_kl > 1.5 * self.target_kl:
                    # early stopping
                    break
            L.log('train_selector/PPO_actor_loss', torch.stack(actor_losses).mean(), step)
            L.log('train_selector/PPO_critic_loss', torch.stack(critic_losses).mean(), step)
            L.log('train_selector/PPO_entropy_loss', torch.stack(ent_losses).mean(), step)
            L.log('train_selector/PPO_loss', torch.stack(losses).mean(), step)
            L.log('train_selector/approx_kl', approx_kl, step)
            L.log('train_selector/old_approx_kl', old_approx_kl, step)
            L.log('train_selector/clip_frac', np.mean(clip_fracs), step)
            L.log('train_selector/selector_std', self.segment_selector.std(step), step)
        buffer.clear()
        if self.use_supervised:
            self.update_supervised(supervised_buffer, L, step)

    def state_dict(self):
        return {
            # 'segment_selector': self.segment_selector.state_dict(),
            # 'critic': self.critic.state_dict(),
            # actor and critic are already loaded
            'optimizer': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        # self.segment_selector.load_state_dict(state_dict['segment_selector'])
        # self.critic.load_state_dict(state_dict['critic'])
        self.optimizer.load_state_dict(state_dict['optimizer'])