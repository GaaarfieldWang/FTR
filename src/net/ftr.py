import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from algorithms.modules import NormalizeImg, Flatten
from algorithms.modules import weight_init
import utils

class RegionEmbedding(nn.Module):
    '''
    map the input image regions to embeddings

    (B, R, C, H, W) -> (B, R, embed_dim)
    '''
    def __init__(self, obs_shape, region_num, in_channels, stack_num, num_layers, num_filters, embed_dim, num_heads):
        super().__init__()
        self.preprocess_layer = nn.Sequential(*[NormalizeImg()])
        if obs_shape[1] == 168:
            kernel_size, stride = 6, 4
            self.current_image_size = obs_shape[1] // 4
        else:
            kernel_size, stride = 3, 2
            self.current_image_size = obs_shape[1] // 2
        self.layers = [nn.Conv2d(in_channels, num_filters, kernel_size, stride=stride, padding=1)]
        self.shape = obs_shape[1:]
        self.region_num = region_num
        self.in_channels = in_channels
        self.stack_num = 1
        self.embed_dim = embed_dim

        for _ in range(1, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.current_image_size = self.current_image_size // 2
        self.layers.append(Flatten())
        out_num = num_filters * self.current_image_size ** 2
        self.layers.append(nn.Linear(out_num, embed_dim))
        self.layers = nn.Sequential(*self.layers)
        self.layers.apply(weight_init)

    def forward(self, x, detach=False):
        '''
        x: torch.Tensor, shape = (batch_size, (region_num + 1) * channels , height, width)
        Last region is the whole frame

        return:
        embeddings: torch.Tensor, shape = (batch_size, region_num + 1, embed_dim),
        batch_size: int,
        masks: torch.Tensor, shape = (B, 1, R)
        '''
        S, R, C, H, W = self.stack_num, self.region_num + 1, self.in_channels, self.shape[0], self.shape[1]
        x = x.reshape(-1, C, H, W)
        B = x.shape[0] // S // R
        x = self.preprocess_layer(x)
        mask = torch.sum(x, dim=(1, 2, 3)).reshape(B * S, 1, -1)[:, :, :-1]
        mask = torch.where(mask != 0, False, True)
        tokens = self.layers(x).reshape(B * S, R, -1)
        assert not torch.isnan(tokens).any(), "Tokens contains NaN values"
        if detach:
            tokens = tokens.detach()
        # avoid the case that all the pixels in the region are masked
        # for i in range(len(mask)):
        #     for j in range(len(mask[i])):
        #         if torch.all(mask[i, j]):
        #             mask[i, j, 0] = False
        all_true = torch.all(mask, dim=-1)
        mask[..., 0][all_true] = False
        return tokens, B, mask
    
class SegmentSelector(nn.Module):
    """ 
    Segment Selector (High Level Selector) in FTR, return the probabilities of selecting each region
    """
    def __init__(self, region_embedding, obs_shape, region_num, in_channels, stack_num, num_layers, num_filters, embed_dim, num_heads, selector_type=2, init_std=0.05, end_std=0.05, std_steps=10000):
        super().__init__()
        self.selector_type = selector_type
        self.region_embedding = region_embedding
        self.shape = obs_shape[1:]
        self.region_num = region_num
        self.in_channels = in_channels
        self.stack_num = 1
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.init_std = init_std
        self.end_std = end_std
        self.std_steps = std_steps

        self.q = nn.Linear(embed_dim, num_heads * embed_dim)
        self.k = nn.Linear(embed_dim, num_heads * embed_dim)
        # --- no 'v' network

        # for continuous action space, depricated
        self.log_std = nn.Parameter(torch.tensor(init_std).log().cuda(), requires_grad=True)
    
    def std(self, step):
        # return self.log_std.exp()
        assert step is not None, "The step should not be None."
        if step > self.std_steps:
            return self.end_std
        else:
            return self.init_std + (self.end_std - self.init_std) * step / self.std_steps

    def forward(self, x, return_logits=False, return_head_logits=False, return_all=False, return_raw_probs=False, detach=False, step=None):
        '''
        x: torch.Tensor, shape = (batch_size, stack_num * (region_num + 1) * channels , height, width)
        Last region is the whole frame

        return: ret_obs, probs
        ret_obs: torch.Tensor, shape = (batch_size, stack_num * channels , height, width)
        probs: torch.Tensor, shape = (B * S, 1, region_num)
        raw_probs: torch.Tensor, shape = (B, S, region_num, 2)
        '''
        if self.selector_type == 0:
            return self.forward_softmax(x, return_logits, return_head_logits, return_all, return_raw_probs, detach)
        S, R, C, H, W = self.stack_num, self.region_num + 1, self.in_channels, self.shape[0], self.shape[1]
        tokens, B, mask = self.region_embedding(x, detach=detach)

        tokens_frame = tokens[:, -1:, :] # o_t
        tokens_segment = tokens[:, :-1, :] # o_t^seg
        q = self.q(tokens_frame).reshape(B * S, 1, self.num_heads, self.embed_dim).transpose(-3, -2)
        k = self.k(tokens_segment).reshape(B * S, R - 1, self.num_heads, self.embed_dim).transpose(-3, -2)
        v = x.reshape(B * S, R, C * H * W)[:, :-1, :].float()

        attention = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(k.shape[-1]) # (B * S, num_heads, 1, R - 1)
        mask = torch.cat([torch.unsqueeze(mask, dim=1)] * self.num_heads, dim=1)    # (B * S, num_heads, 1, R - 1)
        attention = attention.masked_fill_(mask, float("-inf"))
        attention = torch.mean(attention, dim=1) # (B * S, 1, R - 1)
        attention = attention.reshape(B * S, R - 1)
        probs = torch.sigmoid(attention) # (B * S, R - 1)
        raw_probs = probs.clone().reshape(B, S, R - 1, 1)
        probs = probs.reshape(B * S, 1, R - 1)

        raw_probs = torch.cat([1 - raw_probs, raw_probs], dim=-1)

        ret_obs = torch.matmul(probs, v) # (B * S, R - 1, -1)

        # vector 2 image
        ret_obs = ret_obs.reshape(-1, S * C, H, W)

        if return_logits:
            return probs
        # elif return_head_logits:
        #     return multi_probs
        elif return_all:
            return ret_obs, probs
        elif return_raw_probs:
            if step is None:
                return ret_obs, raw_probs
            else:
                return ret_obs, raw_probs, self.std(step)
        else:
            return ret_obs
        
    def _sample_action_from_probs(self, raw_probs, eval_mode=False, step=None):
        '''
        raw_probs: torch.Tensor, shape = (B, S, region_num, 2)
        The last dimension is the probability of taking actions 0 and 1, meaning selecting or not selecting the region

        return: List[bool], shape = (B, S, region_num)
        '''
        if self.selector_type == 0:
            return self._sample_action_from_probs_softmax(raw_probs, eval_mode)
        elif self.selector_type == 2:
            return self._sample_action_from_probs_continuous(raw_probs, eval_mode, step)
        B, S, R, two = raw_probs.shape
        assert two == 2, "The last dimension of raw_probs should be 2."
        # raw_probs = raw_probs.reshape(-1, 2)
        if not eval_mode:
            m = Categorical(raw_probs)
            actions = m.sample()    # (B, S, region_num)
            # actions = actions.reshape(B, S, R)
        else:
            actions = torch.argmax(raw_probs, dim=-1)   # (B, S, region_num)
        actions = actions.tolist()
        return actions

    def _sample_action_from_probs_softmax(self, raw_probs, eval_mode=False):
        '''
        raw_probs: torch.Tensor, shape = (B, S, region_num, 2)
        The last dimension is the probability of taking actions 0 and 1, meaning selecting or not selecting the region

        return: List[bool], shape = (B, S, region_num)
        '''
        B, S, R, two = raw_probs.shape
        assert two == 2, "The last dimension of raw_probs should be 2."
        probs = raw_probs[:, :, :, 1]   # (B, S, region_num)
        if not eval_mode:
            m = Categorical(probs)
            selected = m.sample()    # (B, S)
        else:
            selected = torch.argmax(probs, dim=-1)  # (B, S)
        actions = torch.zeros(B, S, R).to(raw_probs.device)
        # for b in range(B):
        #     for s in range(S):
        #         actions[b, s, selected[b, s]] = 1
        actions = actions.scatter_(2, selected.unsqueeze(-1), 1).type(torch.int64)
        actions = actions.tolist()
        return actions
    
    def _sample_action_from_probs_continuous(self, raw_probs, eval_mode=False, step=None):
        '''
        raw_probs: torch.Tensor, shape = (B, S, region_num, 2)
        The last dimension is the probability of taking actions 0 and 1, meaning selecting or not selecting the region

        return: List[bool], shape = (B, S, region_num)
        '''
        B, S, R, two = raw_probs.shape
        assert two == 2, "The last dimension of raw_probs should be 2."
        probs = raw_probs[:, :, :, 1]   # (B, S, region_num)
        if not eval_mode:
            m = torch.distributions.Normal(probs, self.std(step))
            actions = m.sample()    # (B, S, region_num)
        else:
            actions = probs
        actions = actions.tolist()
        return actions
    
    def forward_softmax(self, x, return_logits=False, return_head_logits=False, return_all=False, return_raw_probs=False, detach=False):
        '''
        x: torch.Tensor, shape = (batch_size, stack_num * (region_num + 1) * channels , height, width)
        Last region is the whole frame

        return: ret_obs, probs
        ret_obs: torch.Tensor, shape = (batch_size, stack_num * channels , height, width)
        probs: torch.Tensor, shape = (B * S, 1, region_num)
        raw_probs: torch.Tensor, shape = (B, S, region_num, 2)
        '''
        S, R, C, H, W = self.stack_num, self.region_num + 1, self.in_channels, self.shape[0], self.shape[1]
        tokens, B, mask = self.region_embedding(x, detach=detach)

        tokens_frame = tokens[:, -1:, :] # o_t
        tokens_segment = tokens[:, :-1, :] # o_t^seg
        q = self.q(tokens_frame).reshape(B * S, 1, self.num_heads, self.embed_dim).transpose(-3, -2)
        k = self.k(tokens_segment).reshape(B * S, R - 1, self.num_heads, self.embed_dim).transpose(-3, -2)
        v = x.reshape(B * S, R, C * H * W)[:, :-1, :].float() 

        attention = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(k.shape[-1]) # (B * S, num_heads, 1, R - 1)
        mask = torch.cat([torch.unsqueeze(mask, dim=1)] * self.num_heads, dim=1)
        attention = attention.masked_fill_(mask, float("-inf"))

        multi_probs = torch.softmax(attention, dim=-1)  # (B * S, num_heads, 1, R - 1)
        probs = torch.mean(multi_probs, dim=1)  # (B * S, 1, R - 1)
        raw_probs = probs.clone().reshape(B, S, R - 1, 1)
        probs = probs.reshape(B * S, 1, R - 1)
        raw_probs = torch.cat([1 - raw_probs, raw_probs], dim=-1)   # (B, S, R - 1, 2)
        ret_obs = torch.matmul(probs, v)

        # vector 2 image
        ret_obs = ret_obs.reshape(-1, S * C, H, W)

        if return_logits:
            return probs
        # elif return_head_logits:
        #     return multi_probs
        elif return_all:
            return ret_obs, probs
        elif return_raw_probs:
            return ret_obs, raw_probs
        else:
            return ret_obs

class ImageSelectorCritic(nn.Module):
    '''
    Value function on the state

    (B, R, C, H, W) -> (B, 1)
    '''
    def __init__(self, region_embedding, obs_shape, region_num, in_channels, stack_num, num_layers, num_filters, embed_dim, num_heads):
        super().__init__()
        self.region_embedding = region_embedding
        self.shape = obs_shape[1:]
        self.region_num = region_num
        self.in_channels = in_channels
        self.stack_num = 1
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.q = nn.Linear(embed_dim, num_heads * embed_dim)
        self.k = nn.Linear(embed_dim, num_heads * embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, x, detach=False):
        '''
        x: torch.Tensor, shape = (batch_size, (region_num + 1) * channels , height, width)
        Last region is the whole frame

        return: values, torch.Tensor, shape = (B, 1)
        '''
        S, R, C, H, W = self.stack_num, self.region_num + 1, self.in_channels, self.shape[0], self.shape[1]
        tokens, B, mask = self.region_embedding(x, detach=detach)

        tokens_frame = tokens[:, -1:, :]
        tokens_segment = tokens[:, :-1, :]
        q = self.q(tokens_frame).reshape(B * S, 1, self.num_heads, self.embed_dim).transpose(-3, -2) # (B * S, num_heads, 1, embed_dim)
        k = self.k(tokens_segment).reshape(B * S, R - 1, self.num_heads, self.embed_dim).transpose(-3, -2) # (B * S, num_heads, R - 1, embed_dim)
        v = self.v(tokens_segment).reshape(B * S, R - 1, self.embed_dim)    # (B * S, R - 1, embed_dim)

        attention = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(k.shape[-1]) # (B * S, num_heads, 1, R - 1)
        mask = torch.cat([torch.unsqueeze(mask, dim=1)] * self.num_heads, dim=1)
        attention = attention.masked_fill_(mask, float("-inf"))

        multi_probs = torch.softmax(attention, dim=-1)
        probs = torch.mean(multi_probs, dim=1)  # (B * S, 1, R - 1)
        atten_out = torch.matmul(probs, v)    # (B * S, 1, embed_dim)
        values = self.linear(atten_out).reshape(B * S, 1)
        return values

class ImageSelectorCriticQsaFunction(nn.Module):
    '''
    Q function on the state and action

    (B, R, C, H, W), (B, R) -> (B, 1)
    '''
    def __init__(self, obs_shape, region_num, in_channels, stack_num, num_layers, num_filters, embed_dim, num_heads):
        super().__init__()
        self.shape = obs_shape[1:]
        self.region_num = region_num
        self.in_channels = in_channels
        self.stack_num = 1
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.q = nn.Linear(embed_dim, num_heads * embed_dim)
        self.k = nn.Linear(embed_dim, num_heads * embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, tokens, mask, action, B, S, R, C, H, W):
        tokens_frame = tokens[:, -1:, :]
        tokens_segment = tokens[:, :-1, :]
        # q = self.q(tokens_frame).reshape(B * S, 1, self.num_heads, self.embed_dim).transpose(-3, -2) # (B * S, num_heads, 1, embed_dim)
        # k = self.k(tokens_segment).reshape(B * S, R - 1, self.num_heads, self.embed_dim).transpose(-3, -2) # (B * S, num_heads, R - 1, embed_dim)
        v = self.v(tokens_segment).reshape(B * S, R - 1, self.embed_dim)    # (B * S, R - 1, embed_dim)

        # attention = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(k.shape[-1]) # (B * S, num_heads, 1, R - 1)
        # mask = torch.cat([torch.unsqueeze(mask, dim=1)] * self.num_heads, dim=1)
        # attention = attention.masked_fill_(mask, float("-inf"))

        # multi_probs = torch.softmax(attention, dim=-1)
        # probs = torch.mean(multi_probs, dim=1)  # (B * S, 1, R - 1)
        # action is used to change the distribution of the values
        action = action.reshape(B * S, 1, R - 1)
        probs = action
        atten_out = torch.matmul(probs, v)    # (B * S, 1, embed_dim)
        values = self.linear(atten_out).reshape(B * S, 1)
        return values

class ImageSelectorCriticQsa(nn.Module):
    '''
    Q function on the state and action

    (B, R, C, H, W), (B, R) -> (B, 1)
    '''
    def __init__(self, region_embedding, obs_shape, region_num, in_channels, stack_num, num_layers, num_filters, embed_dim, num_heads):
        super().__init__()
        self.region_embedding = region_embedding
        self.shape = obs_shape[1:]
        self.region_num = region_num
        self.in_channels = in_channels
        self.stack_num = 1
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.Q1 = ImageSelectorCriticQsaFunction(obs_shape, region_num, in_channels, stack_num, num_layers, num_filters, embed_dim, num_heads)
        self.Q2 = ImageSelectorCriticQsaFunction(obs_shape, region_num, in_channels, stack_num, num_layers, num_filters, embed_dim, num_heads)

    def forward(self, x, action, detach=False):
        '''
        x: torch.Tensor, shape = (batch_size, (region_num + 1) * channels , height, width)
        Last region is the whole frame
        action: torch.Tensor, shape = (batch_size, region_num)

        return: values, torch.Tensor, shape = (B, 1)
        '''
        S, R, C, H, W = self.stack_num, self.region_num + 1, self.in_channels, self.shape[0], self.shape[1]
        tokens, B, mask = self.region_embedding(x, detach=detach)
        Q1 = self.Q1(tokens, mask, action, B, S, R, C, H, W)
        Q2 = self.Q2(tokens, mask, action, B, S, R, C, H, W)
        return Q1, Q2
