import os
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import algorithms.modules as m
from algorithms.drqv2 import DrQV2Agent
from algorithms.high_level_selector_ppo import HighLevelPPO
from algorithms.high_level_selector_sac import HighLevelSAC
from net.sam2model import SAM2Model
from net.ftr import SegmentSelector, RegionEmbedding, ImageSelectorCritic, ImageSelectorCriticQsa
from vlm import VLM_Model


class FTR_DRQ(DrQV2Agent):
    def __init__(self, obs_shape, action_shape, args):
        self.h, self.w = obs_shape[-2], obs_shape[-1]
        self.stack_num = args.frame_stack # default=3
        self.channels = args.channels # default=3
        self.region_num = args.masked_region_num # default=9

        self.sam2_model = SAM2Model(args=args)

        self.region_embedding = RegionEmbedding(obs_shape, args.masked_region_num, args.channels,
                                                       args.frame_stack, args.num_selector_layers, args.num_filters,
                                                       args.embed_dim, args.attention_heads).cuda()

        selector_layers = SegmentSelector(self.region_embedding, obs_shape, args.masked_region_num, args.channels,
                                                       args.frame_stack, args.num_selector_layers, args.num_filters,
                                                       args.embed_dim, args.attention_heads, args.selector_type,
                                                       args.init_std, args.end_std, args.std_steps).cuda()
        if args.sac_selector:
            self.selector_critic = ImageSelectorCriticQsa(self.region_embedding, obs_shape, args.masked_region_num, args.channels,
                                                         args.frame_stack, args.num_selector_layers, args.num_filters,
                                                            args.embed_dim, args.attention_heads).cuda()
        else:
            self.selector_critic = ImageSelectorCritic(self.region_embedding, obs_shape, args.masked_region_num, args.channels,
                                                       args.frame_stack, args.num_selector_layers, args.num_filters,
                                                       args.embed_dim, args.attention_heads).cuda()

        self.complete_selector = selector_layers

        self.high_level_selector = m.HighLevelSelector(args=args, sam2_model=self.sam2_model, segment_selector=self.complete_selector)

        if args.sac_selector:
            self.selector_trainer = HighLevelSAC(args=args, segment_selector=self.complete_selector, critic=self.selector_critic)
        else:
            # train high level selector with ppo
            self.selector_trainer = HighLevelPPO(args=args, segment_selector=self.complete_selector, critic=self.selector_critic)
        
        super(FTR_DRQ, self).__init__(obs_shape, action_shape, args)
        self.train()
        self.critic_target.train()
        self.vlm_model = VLM_Model(args=args)
        self.vlm_info = {
            'segments': [],
            'response': []
        }
    
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.selector_trainer.train(training)
    
    def segment_image(self, obs, env_reset = False):
        return self.high_level_selector.segment_image(obs, env_reset)
    
    def select_image(self, obs):
        with torch.no_grad():
            current_obs = self._obs_to_input(obs)
            # current_obs: torch.Tensor, shape = (1, stack_num * (region_num + 1) * channels, height, width)
            obs, logits = self.complete_selector(current_obs[:, -(self.region_num + 1) * self.channels:], return_all=True)
            selected_obs = torch.squeeze(obs)[-self.channels:].cpu().numpy()
            logits = logits.reshape(-1, self.region_num)[-1].cpu().detach().tolist()
            return logits, np.transpose(selected_obs * 255, (1, 2, 0)).astype(np.uint8)
            
    def select_image_for_plot(self, obs_selected):
        '''
        obs_selected: np.array (stack_num * channels, height, width)

        return: selected_image: np.array (height, width, channels), dtype=np.uint8
        '''
        obs_selected = torch.tensor(obs_selected).reshape(self.stack_num, self.channels, self.h, self.w)
        obs_selected = obs_selected[0]
        return np.transpose(obs_selected.cpu().numpy(), (1, 2, 0)).astype(np.uint8)

    def get_low_level_state(self, obs, env_reset=False, eval_mode=False, step=None):
        '''
        obs: np.array / utils.LazyFrames (stack_num * channels, height, width)
        env_reset: bool, whether the environment is reset

        return: obs_selected, obs_segments, high_level_action
        obs_selected: np.array (stack_num * channels, height, width)
        obs_segments: np.array (stack_num * (region_num + 1) * channels, height, width)
        high_level_action: List[int], meaning the selected region, shape = (stack_num, region_num). If the list is None, it means that it's using camera_predictor
        last_s: np.array ((region_num + 1) * channels, height, width), the last high level state
        last_a: Tuple[torch.Tensor, List[int]], the last high level action, including the probs and high_level_action
        '''
        return self.high_level_selector.get_low_level_state(obs, env_reset, eval_mode, step)
    
    def set_low_level_state(self, obs, masks, high_level_action):
        '''
        obs: np.array / utils.LazyFrames (stack_num * channels, height, width)
        masks: np.array ((region_num + 1) * channels, height, width)
        high_level_action: List[int], meaning the selected region, len = region_num

        return: obs_selected, obs_segments
        obs_selected: np.array (stack_num * channels, height, width)
        obs_segments: np.array (stack_num * (region_num + 1) * channels, height, width)
        '''
        return self.high_level_selector.set_low_level_state(obs, masks, high_level_action)

    def time_to_segment(self):
        return self.high_level_selector.time_to_segment()

    def get_ground_truth_high_level_action(self, obs_segments):
        '''
        obs_segments: np.array ((region_num + 1) * channels, height, width)

        return ground_truth_high_level_action: List[int], meaning the selected region, len = region_num
        '''
        return self._vlm_get_ground_truth_high_level_action(obs_segments)

    def _manually_get_ground_truth_high_level_action(self, obs_segments):
        '''
        obs_segments: np.array ((region_num + 1) * channels, height, width)

        return ground_truth_high_level_action: List[int], meaning the selected region, len = region_num
        '''
        h, w = obs_segments.shape[-2], obs_segments.shape[-1]
        all_segments = torch.tensor(obs_segments).reshape(self.region_num + 1, self.channels, h, w)
        all_segments = all_segments[:-1, :, :, :].cpu().numpy().astype(np.uint8) # (region_num, channels, h, w)
        high_level_action = [0 for _ in range(self.region_num)]
        cur_imgs = all_segments
        cur_imgs = np.transpose(cur_imgs, (0, 2, 3, 1)) # (region_num, h, w, channels)
        # subplot
        fig, axs = plt.subplots(3, 3, figsize=(12, 12))
        for j in range(self.region_num):
            axs[j // 3, j % 3].imshow(cur_imgs[j])
            axs[j // 3, j % 3].set_title(str(j))
            axs[j // 3, j % 3].axis('off')
        plt.savefig('region_imgs.png')
        plt.close()
        str_input = input("Please input the selected regions " + " (0-8): ")
        selected_regions = str_input.split()
        for region in selected_regions:
            high_level_action[int(region)] = 1
        print(f'high_level_action: {high_level_action}')
        return high_level_action

    def _vlm_get_ground_truth_high_level_action(self, obs_segments):
        '''
        obs_segments: np.array ((region_num + 1) * channels, height, width)

        return ground_truth_high_level_action: List[int], meaning the selected region, len = region_num
        '''
        h, w = obs_segments.shape[-2], obs_segments.shape[-1]
        all_segments = torch.tensor(obs_segments).reshape((self.region_num + 1) * self.channels, h, w)
        all_segments = all_segments.cpu().numpy().astype(np.uint8)
        response = self.vlm_model.predict(all_segments)
        self.vlm_info['segments'].append(all_segments)
        self.vlm_info['response'].append(response)
        return response["results"]

    def save_supervised_info(self, work_dir):
        with open(os.path.join(work_dir, 'vlm_info.pkl'), 'wb') as f:
            pickle.dump(self.vlm_info, f)
    
    def update_supervised(self, supervised_buffer, L, step):
        return self.selector_trainer.update_supervised(supervised_buffer, L, step)

    def update_high_level(self, buffer, supervised_buffer, L, step):
        return self.selector_trainer.update(buffer, supervised_buffer, L, step)

    def state_dict(self):
        return {
            'encoder': self.encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'encoder_opt': self.encoder_opt.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'selector': self.complete_selector.state_dict(),
            'selector_critic': self.selector_critic.state_dict(),
            'selector_trainer': self.selector_trainer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict['encoder'])
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        self.encoder_opt.load_state_dict(state_dict['encoder_opt'])
        self.actor_opt.load_state_dict(state_dict['actor_opt'])
        self.critic_opt.load_state_dict(state_dict['critic_opt'])
        # high level selector
        # self.complete_selector.load_state_dict(state_dict['selector'])
        # self.selector_critic.load_state_dict(state_dict['selector_critic'])
        # self.selector_trainer.load_state_dict(state_dict['selector_trainer'])
        # self.selector_trainer.segment_selector = self.complete_selector
        # self.selector_trainer.critic = self.selector_critic