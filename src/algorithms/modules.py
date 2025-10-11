import math
import utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt


def _get_out_shape_cuda(in_shape, layers):
    x = torch.randn(*in_shape).cuda().unsqueeze(0)
    return layers(x).squeeze(0).shape


def _get_out_shape(in_shape, layers):
    x = torch.randn(*in_shape).unsqueeze(0)
    return layers(x).squeeze(0).shape


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability"""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function, see appendix C from https://arxiv.org/pdf/1812.05905.pdf"""
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal distribution, see https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf"""

    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class CenterCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        assert size in {84, 100}, f'unexpected size: {size}'
        self.size = size

    def forward(self, x):
        assert x.ndim == 4, 'input must be a 4D tensor'
        if x.size(2) == self.size and x.size(3) == self.size:
            return x
        assert x.size(3) == 100, f'unexpected size: {x.size(3)}'
        if self.size == 84:
            p = 8
        return x[:, :, p:-p, p:-p]


class NormalizeImg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / 255.


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class RLProjection(nn.Module):
    def __init__(self, in_shape, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(in_shape[0], out_dim),
            nn.LayerNorm(out_dim),
            nn.Tanh()
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.projection(x)


class SharedCNN(nn.Module):
    """
    Obtain an intermediate embedding of the original image; 
    the output shape is (-1, out_channel=num_filters, W, H).
    """

    def __init__(self, obs_shape, num_layers=11, num_filters=32):
        super().__init__()
        assert len(obs_shape) == 3
        self.num_layers = num_layers
        self.num_filters = num_filters

        self.layers = [CenterCrop(size=84), NormalizeImg(), nn.Conv2d(
            obs_shape[0], num_filters, 3, stride=2)]
        for _ in range(1, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(
                nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(obs_shape, self.layers)
        self.apply(weight_init)

    def forward(self, x):
        return self.layers(x)


class HeadCNN(nn.Module):
    """
    Obtain the flattened final embedding based on the SharedCNN.
    """

    def __init__(self, in_shape, num_layers=0, num_filters=32):
        super().__init__()
        self.layers = []
        for _ in range(0, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(
                nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers.append(Flatten())
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(in_shape, self.layers)
        self.apply(weight_init)

    def forward(self, x):
        return self.layers(x)


class SelectorCNN(nn.Module):
    def __init__(self, selector_layers, obs_shape, region_num=5, in_channels=3, stack_num=3, num_shared_layers=11,
                 num_filters=32):
        super().__init__()
        assert len(obs_shape) == 3
        # assert region_num * in_channels * stack_num == obs_shape[0]
        self.obs_shape = obs_shape
        self.in_channels = in_channels
        self.stack_num = stack_num
        self.num_filters = num_filters

        self.selector_layers = selector_layers

        self.shared_layers = [
            nn.Conv2d(self.stack_num * self.in_channels, num_filters, 3, stride=2)]
        for _ in range(1, num_shared_layers):
            self.shared_layers.append(nn.ReLU())
            self.shared_layers.append(
                nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.shared_layers = nn.Sequential(*self.shared_layers)

        self.out_shape = _get_out_shape([self.stack_num * self.in_channels, self.obs_shape[-2], self.obs_shape[-1]],
                                        self.shared_layers)
        self.shared_layers.apply(weight_init)

    def forward(self, x):
        x = self.selector_layers(x)
        x = self.shared_layers(x)

        return x


class Encoder(nn.Module):
    def __init__(self, shared_cnn, head_cnn, projection):
        super().__init__()
        self.shared_cnn = shared_cnn
        self.head_cnn = head_cnn
        self.projection = projection
        self.out_dim = projection.out_dim

    def forward(self, x, detach=False):
        x = self.shared_cnn(x)
        x = self.head_cnn(x)
        if detach:
            x = x.detach()
        return self.projection(x)


class Actor(nn.Module):
    def __init__(self, encoder, action_shape, hidden_dim, log_std_min, log_std_max):
        super().__init__()
        self.encoder = encoder
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )
        self.mlp.apply(weight_init)

    def forward(self, x, compute_pi=True, compute_log_pi=True, detach=False):
        x = self.encoder(x, detach)
        mu, log_std = self.mlp(x).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        return self.trunk(torch.cat([obs, action], dim=1))


class Critic(nn.Module):
    def __init__(self, encoder, action_shape, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.Q1 = QFunction(
            self.encoder.out_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.out_dim, action_shape[0], hidden_dim
        )

    def forward(self, x, action, detach=False):
        x = self.encoder(x, detach)
        return self.Q1(x, action), self.Q2(x, action)
    
class ActorCritic(nn.Module):
    """An actor-critic network for parsing parameters.

    Using ``actor_critic.parameters()`` instead of set.union or list+list

    :param nn.Module actor: the actor network.
    :param nn.Module critic: the critic network.
    """

    def __init__(self, actor: nn.Module, critic: nn.Module) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic

class SiameseNet(nn.Module):
    def __init__(self, input_dim, hidden_sizes, noise_scale=0.05):
        super(SiameseNet, self).__init__()
        assert len(hidden_sizes) > 1
        temp = input_dim
        output_dim = hidden_sizes[-1]

        fc_layers = []
        for hidden_size in hidden_sizes[:-1]:
            fc_layer = nn.Linear(input_dim, hidden_size)
            nn.init.orthogonal_(fc_layer.weight.data)
            fc_layer.bias.data.fill_(0.)

            fc_layers.append(fc_layer)
            fc_layers.append(nn.ReLU())

            input_dim = hidden_size

        last_layer = nn.Linear(hidden_sizes[-2], output_dim)
        nn.init.orthogonal_(fc_layer.weight.data)
        last_layer.bias.data.fill_(0.)

        fc_layers.append(last_layer)

        self.fc_layers = nn.Sequential(*fc_layers)

        self.noise_scale = noise_scale

    def forward(self, h, a):
        noise = torch.randn_like(h) * self.noise_scale
        h = h + noise
        h = h.clamp(-1.0, 1.0)
        noise = torch.randn_like(a) * self.noise_scale
        a = (a + noise).clamp(-1.0, 1.0)
        x = torch.cat((h, a), dim=1)
        return self.fc_layers(x)


class SiameseNet_a(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_sizes, max_action):
        super(SiameseNet_a, self).__init__()
        output_dim = hidden_sizes[-1]

        fc_layers = []
        for hidden_size in hidden_sizes:
            fc_layer = nn.Linear(input_dim, hidden_size)
            nn.init.orthogonal_(fc_layer.weight.data)
            fc_layer.bias.data.fill_(0.)

            fc_layers.append(fc_layer)
            fc_layers.append(nn.ReLU())

            input_dim = hidden_size

        last_layer = nn.Linear(hidden_sizes[-1], action_dim)
        nn.init.orthogonal_(fc_layer.weight.data)
        last_layer.bias.data.fill_(0.)

        fc_layers.append(last_layer)

        self.fc_layers = nn.Sequential(*fc_layers)

        self.max_action = max_action

    def forward(self, h1, h2):
        h = torch.cat([h1 + h2, torch.abs(h1 - h2)], dim=1)
        h = self.fc_layers(h)
        return self.max_action * torch.tanh(h)


def make_siamese_net(type, feature_dim, action_dim, hidden_sizes, noise_scale=0.0, max_action=1.0):
    if type == 'critic':
        return SiameseNet(feature_dim + action_dim, hidden_sizes, noise_scale=noise_scale)
    else:
        return SiameseNet_a(feature_dim * 2, action_dim, hidden_sizes, max_action)

def clip_loss(probs, target, margin=0.1):
    assert probs.size() == target.size()
    loss = torch.mean(
        target * torch.relu(0.5 + margin - probs) +
        (1 - target) * torch.relu(probs - 0.5 + margin)
    )
    return loss

class HighLevelSelector:
    def __init__(self, args, sam2_model, segment_selector):
        self.args = args
        self.use_selector = args.use_selector
        self.sam2_model = sam2_model
        self.segment_selector = segment_selector
        self.add_original_frame = args.add_original_frame
        self.color_type = args.color_type
        self.image_crop_size = args.image_crop_size
        self.stack_num = args.frame_stack # default=3
        self.channels = args.channels # default=3
        self.region_num = args.masked_region_num # default=9
        self.attention_heads = args.attention_heads # default=4
        self.frame_idx = 0
        self.timer = 0
        self.segment_interval = args.segment_interval
        self.last_segments = None
        self.last_s_h = None
        self.last_a_h = None

    def _obs_to_input_3d(self, obs):
        if isinstance(obs, utils.LazyFrames):
            _obs = np.array(obs)
        else:
            _obs = obs
        if not isinstance(_obs,torch.Tensor):
            _obs = torch.FloatTensor(_obs).cuda()
        else:
            _obs = _obs.cuda()
        return _obs
    
    def _sample_action_from_probs(self, raw_probs, eval_mode=False, step=None):
        '''
        raw_probs: torch.Tensor, shape = (B, 1, region_num, 2)
        The last dimension is the probability of taking actions 0 and 1, meaning selecting or not selecting the region

        return: List[bool], shape = (B, 1, region_num)
        '''
        return self.segment_selector._sample_action_from_probs(raw_probs, eval_mode, step)

    def _segment_image_sam(self, obs):
        '''
        obs: torch.Tensor, shape = (channels, height, width)

        return:obs_segments, first_frame, cur_masks
        obs_segments: np.array ((region_num + 1) * channels, height, width)
        first_frame: np.array (height, width, channels)
        cur_masks: np.array, shape = (region_num, h, w), dtype = np.bool8
        '''
        h, w = obs.shape[1], obs.shape[2]
        obs_segments = []
        cur_masks = None
        with torch.no_grad():
            cur_obs_segments, cur_masks = self.sam2_model.segment_image(obs)
            
            obs_segments.append(torch.tensor(cur_obs_segments))
            obs_segments = torch.stack(obs_segments)
            obs_segments = obs_segments.reshape((self.region_num + 1) * self.channels, h, w)
            first_frame = obs
            first_frame = first_frame.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        return obs_segments.cpu().numpy().astype(np.uint8), first_frame, cur_masks
    
    def _masks_to_segments(self, image, masks):
        '''
        image: np.array, shape = (height, width, channels), dtype=np.uint8
        masks: np.array, shape = (region_num + add_original_frame, height, width), dtype=np.bool8

        return:np.array, shape = ((region_num + add_original_frame) * channels, height, width), dtype=np.uint8
        '''
        if self.color_type == "rgb":
            total_masks = np.repeat(masks[:, None, :, :], 3, axis=1)
            image_transposed = image.transpose((2, 0, 1))[None, ...]  # (1, 3, H, W)
            total_images = np.repeat(image_transposed, len(masks), axis=0)
            masked_images = (total_masks * total_images).reshape(-1, self.image_crop_size, self.image_crop_size)
        elif self.color_type == "gray":
            total_masks = masks
            gray_image = image.mean(axis=2).astype("uint8")
            total_images = np.repeat(gray_image[None, ...], len(masks), axis=0)  # (N, H, W)
            masked_images = total_masks * total_images
        
        return masked_images.astype(np.uint8)
    
    def _select_regions(self, all_segments, high_level_action):
        '''
        all_segments: torch.Tensor, shape = (stack_num, region_num, channels, height, width)
        high_level_action: List[int], meaning the selected region, shape = (stack_num, region_num)

        return: obs_selected: np.array (stack_num * channels, height, width)
        '''
        h, w = all_segments.shape[3], all_segments.shape[4]
        mask = (high_level_action >= 0.5 if isinstance(high_level_action, torch.Tensor)
                else torch.tensor(high_level_action) >= 0.5)
        # [stack, region] -> [stack, region, 1, 1, 1]
        mask = mask.view(*mask.shape, 1, 1, 1).to(all_segments.device)
        masked_segments = torch.where(
            mask,
            all_segments.float(),
            torch.tensor(-torch.inf, device=all_segments.device)
        )
        obs_selected = masked_segments.max(dim=1).values  # [stack, C, H, W]
        obs_selected = torch.nan_to_num(obs_selected, nan=0.0, posinf=255.0, neginf=0.0)
        return obs_selected.reshape(-1, h, w).clamp(0, 255).cpu().numpy().astype(np.uint8)

    def segment_image(self, obs, env_reset):
        '''
        deprecated!
        obs:np.array / utils.LazyFrames (stack_num * channels, height, width)

        return:obs_segments: np.array (stack_num * (region_num + add_original_frame) * channels, height, width)
        '''
        raise NotImplementedError
        
    def get_low_level_state(self, obs, env_reset=False, eval_mode=False, step=None):
        if not self.use_selector:
            return self.get_low_level_state_dummy(obs, env_reset)
        '''
        obs: np.array / utils.LazyFrames (stack_num * channels, height, width)
        env_reset: bool, whether the environment is reset

        return: obs_selected, obs_segments, high_level_action
        obs_selected: np.array (stack_num * channels, height, width)
        obs_segments: np.array (stack_num * (region_num + 1) * channels, height, width)
        high_level_action: List[int], meaning the selected region, len = region_num. If the list is None, it means that it's using tracking_model
        last_s: np.array ((region_num + 1) * channels, height, width), the last high level state
        last_a: Tuple[torch.Tensor, List[int]], the last high level action, including the probs and high_level_action
        '''
        obs = self._obs_to_input_3d(obs)
        h, w = obs.shape[1], obs.shape[2]
        obs = obs.reshape(self.stack_num, self.channels, h, w)
        self.timer += 1
        if self.timer % self.segment_interval == 0 or env_reset:
            # use segment_model
            self.timer = 0
            obs_segments = []
            first_obs_segments, first_frame, cur_masks = self._segment_image_sam(obs[0])
            obs_segments.append(first_obs_segments)
            obs_segments_torch = torch.tensor(first_obs_segments).reshape(1, (self.region_num + 1) * self.channels, h, w).cuda()
            # high_level_selector
            _, raw_probs = self.segment_selector(obs_segments_torch, return_raw_probs=True)

            high_level_action = self._sample_action_from_probs(raw_probs, eval_mode, step)[0][0]
            self.last_s_h, self.last_a_h = first_obs_segments, (raw_probs, high_level_action)
            # reset tracking_model
            # choose the last stacked frame as the initial frame of camera predictor
            selected_regions = []
            use_white_mask = False
            for i, select in enumerate(high_level_action):
                if select >= 0.5:
                    selected_regions.append(cur_masks[i])
            if len(selected_regions) == 0:
                print('No mask selected, use the white mask')
                print('raw_probs:', [round(p, 3) for p in raw_probs[0, -1, :, 1].cpu().detach().tolist()])
                # plot the first frame and masks
                plt.figure()
                plt.imshow(first_frame)
                plt.savefig("./figures/first_frame.jpg")
                plt.close()
                fig, ax = plt.subplots(3, 3)
                for i in range(self.region_num):
                    ax[i // 3, i % 3].imshow(first_obs_segments[i * 3: (i + 1) * 3].transpose(1, 2, 0))
                    ax[i // 3, i % 3].axis('off')
                    ax[i // 3, i % 3].set_title(f"{raw_probs[0, -1, i, 1].cpu().detach().item():.2f}")
                plt.savefig("./figures/masks.jpg")
                plt.close()
                # end of plot
                selected_regions.append(np.ones_like(cur_masks[0]))
                use_white_mask = True
                self.timer = self.segment_interval - 1 # force to use segment_model at next timestep
            if not use_white_mask:
                self.sam2_model.load_first_frame(first_frame)
                self.frame_idx = 0
                out_frame_idx, out_obj_ids, out_video_res_masks = self.sam2_model.add_new_masks(self.frame_idx, list(range(len(selected_regions))), selected_regions)
                for i in range(1, self.stack_num):
                    self.frame_idx  = self.frame_idx + 1
                    cur_frame = obs[i].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                    # use tracking_model.track to get new mask
                    _, cur_masks = self.sam2_model.track(cur_frame)
                    if len(cur_masks) < self.region_num:
                        # zero padding
                        cur_masks = np.concatenate([cur_masks, np.zeros((self.region_num - len(cur_masks), h, w), dtype=cur_masks[0].dtype)], axis=0)
                    if self.add_original_frame:
                        white_mask = np.ones((h, w), dtype=cur_masks.dtype)
                        cur_masks = np.concatenate([cur_masks, white_mask[np.newaxis, :, :]], axis=0)
                    obs_segments.append(self._masks_to_segments(cur_frame, cur_masks))
            else:
                for i in range(1, self.stack_num):
                    obs_segment = np.zeros((self.region_num + 1, self.channels, h, w), dtype=np.uint8)
                    obs_segment[0] = obs[i].cpu().numpy().astype(np.uint8)
                    obs_segment[-1] = obs[i].cpu().numpy().astype(np.uint8)
                    obs_segment = obs_segment.reshape((self.region_num + 1) * self.channels, h, w)
                    obs_segments.append(obs_segment)
            obs_segments = np.stack(obs_segments)
            self.last_segments = obs_segments[1:]
            obs_segments = obs_segments.reshape(self.stack_num * (self.region_num + 1) * self.channels, h, w)
            all_segments = torch.tensor(obs_segments).reshape(self.stack_num, self.region_num + 1, self.channels, h, w)
            all_segments = all_segments[:, :-1, :, :, :]
            all_high_level_action = np.ones((self.stack_num - 1, self.region_num), dtype=np.uint8).tolist()
            all_high_level_action = [high_level_action] + [[int(action) for action in actions] for actions in all_high_level_action]
            obs_selected = self._select_regions(all_segments, all_high_level_action)
            if use_white_mask:
                obs_selected[:self.channels, :, :] = first_frame.transpose(2, 0, 1)
            return obs_selected, obs_segments, high_level_action, self.last_s_h, self.last_a_h
        else:
            # use tracking_model
            obs_segments = []
            for i in range(self.stack_num):
                if i + 1 < self.stack_num:
                    obs_segments.append(self.last_segments[i])
                else:
                    self.frame_idx  = self.frame_idx + 1
                    cur_frame = obs[i].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                    _, cur_masks = self.sam2_model.track(cur_frame)
                    if len(cur_masks) < self.region_num:
                        cur_masks = np.concatenate([cur_masks, np.zeros((self.region_num - len(cur_masks), h, w), dtype=cur_masks[0].dtype)], axis=0)
                    if self.add_original_frame:
                        white_mask = np.ones((h, w), dtype=cur_masks.dtype)
                        cur_masks = np.concatenate([cur_masks, white_mask[np.newaxis, :, :]], axis=0)
                    obs_segments.append(self._masks_to_segments(cur_frame, cur_masks))
            obs_segments = np.stack(obs_segments)
            self.last_segments = obs_segments[1:]
            obs_segments = obs_segments.reshape(self.stack_num * (self.region_num + 1) * self.channels, h, w)
            all_segments = torch.tensor(obs_segments).reshape(self.stack_num, self.region_num + 1, self.channels, h, w)
            all_segments = all_segments[:, :-1, :, :, :]
            high_level_action = np.ones((self.stack_num, self.region_num), dtype=np.uint8).tolist()
            high_level_action = [[int(action) for action in actions] for actions in high_level_action]
            obs_selected = self._select_regions(all_segments, high_level_action)
            return obs_selected, obs_segments, None, self.last_s_h, self.last_a_h

    def get_low_level_state_dummy(self, obs, env_reset=False):
        '''
        obs: np.array / utils.LazyFrames (stack_num * channels, height, width)
        env_reset: bool, whether the environment is reset

        return: obs_selected, obs_segments, high_level_action
        obs_selected: np.array (stack_num * channels, height, width)
        obs_segments: np.array (stack_num * (region_num + 1) * channels, height, width)
        high_level_action: List[int], meaning the selected region, len = region_num. If the list is None, it means that it's using tracking_model
        '''
        obs = self._obs_to_input_3d(obs)
        h, w = obs.shape[1], obs.shape[2]
        obs = obs.reshape(self.stack_num, self.channels, h, w)
        self.timer += 1
        if self.timer % self.segment_interval == 0 or env_reset:
            self.timer = 0
        obs_selected = obs.reshape(self.stack_num * self.channels, h, w).cpu().numpy().astype(np.uint8)
        obs_segments = []
        for i in range(self.stack_num):
            obs_segment = np.zeros((self.region_num + 1, self.channels, h, w), dtype=np.uint8)
            obs_segment[0] = obs[i].cpu().numpy().astype(np.uint8)
            obs_segment[-1] = obs[i].cpu().numpy().astype(np.uint8)
            obs_segment = obs_segment.reshape((self.region_num + 1) * self.channels, h, w)
            obs_segments.append(obs_segment)
        obs_segments = np.stack(obs_segments)
        obs_segments = obs_segments.reshape(self.stack_num * (self.region_num + 1) * self.channels, h, w)
        high_level_action = [0 for _ in range(self.region_num)]
        return obs_selected, obs_segments, high_level_action, obs_segments[:(self.region_num + 1) * self.channels], (torch.zeros(1, 1, self.region_num, 2), high_level_action)
        
    def set_low_level_state(self, obs, masks, high_level_action):
        '''
        obs: np.array / utils.LazyFrames (stack_num * channels, height, width)
        masks: np.array ((region_num + 1) * channels, height, width)
        high_level_action: List[int], meaning the selected region, len = region_num
        '''
        obs = self._obs_to_input_3d(obs)
        h, w = obs.shape[1], obs.shape[2]
        obs = obs.reshape(self.stack_num, self.channels, h, w)
        self.timer = 0
        obs_segments = []
        obs_segments.append(masks)
        first_frame = obs[0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        cur_masks = masks[:self.region_num * self.channels].reshape(self.region_num, self.channels, h, w)
        cur_masks = cur_masks.sum(axis=1)   # shape = (region_num, h, w)
        cur_masks = cur_masks.astype(np.bool8)
        selected_regions = []
        use_white_mask = False
        for i, select in enumerate(high_level_action):
            if select >= 0.5:
                selected_regions.append(cur_masks[i])
        if len(selected_regions) == 0:
            print('No mask selected, use the white mask')
            selected_regions.append(np.ones_like(cur_masks[0]))
            use_white_mask = True
        if not use_white_mask:
            self.sam2_model.load_first_frame(first_frame)
            self.frame_idx = 0
            out_frame_idx, out_obj_ids, out_video_res_masks = self.sam2_model.add_new_masks(self.frame_idx, list(range(len(selected_regions))), selected_regions)
            for i in range(1, self.stack_num):
                self.frame_idx  = self.frame_idx + 1
                cur_frame = obs[i].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                _, cur_masks = self.sam2_model.track(cur_frame)
                if len(cur_masks) < self.region_num:
                    cur_masks = np.concatenate([cur_masks, np.zeros((self.region_num - len(cur_masks), h, w), dtype=cur_masks[0].dtype)], axis=0)
                if self.add_original_frame:
                    white_mask = np.ones((h, w), dtype=cur_masks.dtype)
                    cur_masks = np.concatenate([cur_masks, white_mask[np.newaxis, :, :]], axis=0)
                obs_segments.append(self._masks_to_segments(cur_frame, cur_masks))
        else:
            for i in range(1, self.stack_num):
                obs_segment = np.zeros((self.region_num + 1, self.channels, h, w), dtype=np.uint8)
                obs_segment[0] = obs[i].cpu().numpy().astype(np.uint8)
                obs_segment[-1] = obs[i].cpu().numpy().astype(np.uint8)
                obs_segment = obs_segment.reshape((self.region_num + 1) * self.channels, h, w)
                obs_segments.append(obs_segment)
        obs_segments = np.stack(obs_segments)
        self.last_segments = obs_segments[1:]
        obs_segments = obs_segments.reshape(self.stack_num * (self.region_num + 1) * self.channels, h, w)
        all_segments = torch.tensor(obs_segments).reshape(self.stack_num, self.region_num + 1, self.channels, h, w)
        all_segments = all_segments[:, :-1, :, :, :]
        all_high_level_action = np.ones((self.stack_num - 1, self.region_num), dtype=np.uint8).tolist()
        all_high_level_action = [high_level_action] + [[int(action) for action in actions] for actions in all_high_level_action]
        obs_selected = self._select_regions(all_segments, all_high_level_action)
        if use_white_mask:
            obs_selected[:self.channels, :, :] = first_frame.transpose(2, 0, 1)
        return obs_selected, obs_segments
    
    def time_to_segment(self):
        return (self.timer + 1) % self.segment_interval == 0
