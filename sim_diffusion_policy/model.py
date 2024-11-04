from typing import Union
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import reduce
import numpy as np
import torch.nn.functional as F
import einops
import math
from typing import Dict, Callable
from torch.nn.modules.batchnorm import _BatchNorm
import torchvision.models as models
import cv2
from torch.autograd.functional import jacobian
from typing import Union
import pytorch3d.transforms as pt
import functools



class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    def __init__(self, 
            from_rep='axis_angle', 
            to_rep='rotation_6d', 
            from_convention=None,
            to_convention=None):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == 'euler_angles':
            assert from_convention is not None
        if to_rep == 'euler_angles':
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != 'matrix':
            funcs = [
                getattr(pt, f'{from_rep}_to_matrix'),
                getattr(pt, f'matrix_to_{from_rep}')
            ]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention) 
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != 'matrix':
            funcs = [
                getattr(pt, f'matrix_to_{to_rep}'),
                getattr(pt, f'{to_rep}_to_matrix')
            ]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention) 
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])
        
        inverse_funcs = inverse_funcs[::-1]
        
        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y
        
    def forward(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)
    
    def inverse(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)
    
# diffusion_policy/diffusion_policy/model/vision/multi_image_obs_encoder.py
# trace/tbsim/models/base_models.py
# trace/tbsim/models/trace_helpers.py
class SingleObEncoder(nn.Module):
    def __init__(self, cfg):
        super(SingleObEncoder, self).__init__()
        self.cfg = cfg
        self.obs_encoder = models.resnet18(pretrained=False)
        self.obs_encoder.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
        input_image_shape = (4, 240, 320)
        out_h = int(math.ceil(input_image_shape[1] / 32.))
        out_w = int(math.ceil(input_image_shape[2] / 32.))
        self.conv_out_shape = (512, out_h, out_w)
        self.obs_encoder.avgpool = SpatialSoftmax(input_shape=self.conv_out_shape)
        self.obs_encoder.fc = nn.Identity()
        self.ee_encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, inputs):
        
        ob_front, ee_pose = inputs

        batch_size = ob_front.shape[0]
        features = []
        for t in range(ob_front.shape[1]):
            rgbd_image_t = ob_front[:, t, :, :, :]  # (batch_size, C, W, H)
            ee_pose_t = ee_pose[:, t, :]  # (batch_size, action_dimension)

            obs_feature = self.obs_encoder(rgbd_image_t)  # (batch_size, 512)
            ee_feature = self.ee_encoder(ee_pose_t)  # (batch_size, 32)
            combined_feature = torch.cat((obs_feature, ee_feature), dim=1)  # (batch_size, 512 + 32)
            features.append(combined_feature)
            features

        features = torch.stack(features, dim=1)  # (batch_size, trajectory_length, 512 + 32)
        ob_embeddings = features.view(batch_size, -1) 
        return ob_embeddings

# diffusion_policy/diffusion_policy/workspace/train_diffusion_unet_image_workspace.py
# diffusion_policy/diffusion_policy/policy/diffusion_unet_image_policy.py
class DiffusionPolicy(nn.Module):
    def __init__(self, 
            cfg,
            obs_encoder,
            noise_scheduler,
            #================#
            action_dim=9,
            obs_feature_dim=512 + 32, # 512+32
            input_dim=521+32+9
        ):
        super().__init__()

        input_dim = action_dim
        global_cond_dim=obs_feature_dim*cfg['n_obs_steps']
        model = ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=128,
            down_dims=(512, 1024, 2048),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True
        )
        mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=cfg["n_obs_steps"],
            fix_obs_steps=True,
            action_visible=False
        )

        # define for training
        self.obs_encoder = obs_encoder
        self.multi_img = cfg['multi_img']
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = mask_generator

        # define parameters
        self.horizon = cfg["horizon"]
        self.n_action_steps = cfg["n_action_steps"]
        self.n_obs_steps = cfg["n_obs_steps"]
        self.num_inference_steps = cfg["num_inference_steps"]
        self.device = cfg["training"]["device"]

        self.guidance = cfg["guidance"]
        self.goal_guided_mode = cfg["testing"]["goal_guided_mode"]
        self.spillage_guided_mode = cfg["testing"]["spillage_guided_mode"]
        self.guided_weight = cfg["testing"]["guided_weight"]
        self.guided_alpha = cfg["testing"]["alpha"]
        self.guided_beta = cfg["testing"]["beta"]
        self.guided_gamma = cfg["testing"]["gamma"]
        self.trigger = False

        self.action_dim = action_dim
        self.input_dim = input_dim
        self.obs_feature_dim = obs_feature_dim

        # normalization
        input_range = torch.load(cfg["data"]["rootPath"] + '/input_range.pt')
        self.input_max = input_range[0,:]
        self.input_min = input_range[1,:]
        self.input_mean = input_range[2,:]
        self.cfg = cfg
        
    
    # ========= training  ============
    def compute_loss(self, inputs):   

        # normalize input
        obs, traj = inputs
        cond_data = traj

        # encode image (obs: [B, H, 4, 256, 256])
        obs_in = obs[:,:self.n_obs_steps,...]
        ee_in = traj[:,:self.n_obs_steps,...]
        obs_features = self.obs_encoder((obs_in, ee_in))
        # obs_features = self.obs_encoder(obs_in.reshape(-1,*obs.shape[2:]))
        global_cond = obs_features.reshape(obs.shape[0], -1)

        # generate inpainting mask
        condition_mask = self.mask_generator(traj.shape)
        loss_mask = ~condition_mask

        # forward diffusion process
        noise = torch.randn(traj.shape, device=traj.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (traj.shape[0],), device=traj.device).long()
        noisy_traj = self.noise_scheduler.add_noise(traj, noise, timesteps)

        # apply conditioning
        noisy_traj[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_traj, timesteps, global_cond=global_cond)
        loss = F.mse_loss(pred, noise, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss

    # ========= inference  ============
    # def goal_objective(self, traj, goal_pos):

    #     # transfer generated ee traj to spoon traj
    #     unnorm_traj = _denormalize(traj, self.input_max, self.input_min, self.input_mean)
    #     spoon_traj = _normalize(from_ee_to_spoon(unnorm_traj), self.input_max, self.input_min, self.input_mean)

    #     goal_pos_xy=goal_pos[:, 0:2]
    #     dist = torch.norm(spoon_traj[:, :, 0:2] - goal_pos_xy[:, None], dim=-1) # (B, 16, 3) - (B, 1, 3) = (B, 16)

    #     if self.goal_guided_mode=='sum_all_frame': # naive method
    #         loss = dist.sum()

    #     elif self.goal_guided_mode=='sum_all_frame_exp': # Scene-Diffuser
    #         loss = torch.exp(1 / dist.clamp(min=0.01)).sum()

    #     elif self.goal_guided_mode=="sum_all_frame_weighted": # Trace and pace
    #         loss_weighting = F.softmin(dist, dim=-1)
    #         loss = loss_weighting * torch.sum((spoon_traj[:, :, 0:2] - goal_pos_xy[:, None])**2, dim=-1)
    #         loss = torch.mean(loss, dim=-1)

    #     # calculate gradient
    #     guided_grad = torch.autograd.grad(loss, traj)[0]
    #     goal_gradient = torch.clip(guided_grad[..., 0:2], min=-0.2, max=0.2)

    #     return goal_gradient

    def local_goal_objective(self, traj, goal_pose):

        # transfer generated ee traj to spoon traj
        unnorm_traj = _denormalize(traj, self.input_max, self.input_min, self.input_mean)
        spoon_traj = _normalize(from_ee_to_spoon(unnorm_traj), self.input_max, self.input_min, self.input_mean)
        action_dim = 2
        goal_pos_xy = goal_pose[:, :action_dim]
        dist = torch.norm(traj[:, :, :action_dim] - goal_pos_xy, dim=-1) # [1, 16]
        
        # local weight
        loss_weighting = F.softmin(dist, dim=-1)
        loss = loss_weighting * torch.sum((traj[:, :, :action_dim] - goal_pos_xy)**2, dim=-1)
        goal_loss = torch.mean(loss, dim=-1)
        print("goal_loss= ", goal_loss)
        return torch.autograd.grad(goal_loss, traj)[0]
    
    def global_goal_objective(self, traj, goal_pose):

        # transfer generated ee traj to spoon traj
        unnorm_traj = _denormalize(traj, self.input_max, self.input_min, self.input_mean)
        spoon_traj = _normalize(from_ee_to_spoon(unnorm_traj), self.input_max, self.input_min, self.input_mean)
        action_dim = 2
        goal_pos_xy = goal_pose[:, :action_dim]

        # global weight
        threshold = 0.4
        d_current = torch.norm(spoon_traj[:, -1, :action_dim] - goal_pos_xy, dim=-1)
        u = torch.min(torch.tensor(1.0, device=self.device), (d_current - threshold) / d_current)
        d_goal = u * d_current
        d_progress = torch.norm(spoon_traj[:, 0, :action_dim] - goal_pos_xy, dim=-1) - torch.norm(spoon_traj[:, -1, :action_dim] - goal_pos_xy, dim=-1)
        goal_loss = F.relu(d_goal - torch.mean(d_progress))
        print("goal_loss= ", goal_loss)
        return torch.autograd.grad(goal_loss, traj)[0]
    
    def spillage_objective(self, obs_in, traj, spillage_classifier):  

        spillage_pred = spillage_classifier(obs_in, traj) # [B, H, 5]

        if self.spillage_guided_mode=='mean':
            prob = spillage_pred[:, :, 0].mean()
        elif self.spillage_guided_mode=='sum':
            prob = spillage_pred[:, :, 0].sum()

        # calculate gradient
        guided_grad = torch.autograd.grad(prob, traj)[0]  
        spillage_gradient = torch.clip(guided_grad, min=-0.2, max=0.2)

        # return spillage_gradient
        return 0

    def amount_objective(self, obs_in, traj, amount_predictor):
        # return amount_gradient
        return 0

    def conditional_sample(self, 
            condition_data, condition_mask,
            global_cond=None,
            goal_position=None,
            spillage_classifier=None,
            obs_in=None,
            start_guidance=False,
            generator=None,
            ):
        
        model = self.model
        scheduler = self.noise_scheduler

        traj = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        if start_guidance:

            traj.requires_grad=True
            for t in scheduler.timesteps:
                traj[condition_mask] = condition_data[condition_mask]
                model_output = model(traj, t, global_cond=global_cond)

                # Compute gradient
                # with torch.enable_grad():
                #     model_output = model_output.detach().requires_grad_(True)
                #     goal_gradient = self.local_goal_objective(model_output, goal_position)       

                # update
                # model_output[..., 0:2] -= 2. * goal_gradient[..., 0:2]
                # # model_output[..., 0:2] -= self.guided_weight * self.guided_alpha * goal_gradient
                # # model_output -= self.guided_weight * (self.guided_beta * spillage_gradient + self.guided_gamma * amount_gradient)
                # # model_output -= self.guided_weight * guided_grad * (1 - scheduler.alphas_cumprod[t]).sqrt()

                traj = scheduler.step(
                    model_output, t, traj, 
                    generator=generator
                    ).prev_sample       

            # optimization
            goal_pos_xy = goal_position[:, :2]
            dist = torch.norm(traj[:, :, :2] - goal_pos_xy, dim=-1)
            print("dist= ", dist)
            dist_in_threshold = torch.where(dist<0.4)
            if len(dist_in_threshold[0])>0:
                self.trigger = True # trigger local weight

            for _ in range(32):
                with torch.enable_grad():
                    traj = traj.detach().requires_grad_(True)
                    traj = traj.to("cuda:0")
                    goal_position = goal_position.to("cuda:0")
                    if self.trigger:
                        grad_weight = 2.
                        guided_grad = self.local_goal_objective(traj, goal_position)
                        print("Local.")
                    else:
                        grad_weight = 0.1
                        guided_grad = self.global_goal_objective(traj, goal_position)
                        print("Global.")

                    traj_clone = traj.clone()
                    traj_clone[..., :2] -= grad_weight * guided_grad[..., :2]
                    traj = traj_clone

        else:     
            for t in scheduler.timesteps:
                # 1. apply conditioning
                traj[condition_mask] = condition_data[condition_mask]
                # 2. predict model output
                model_output = model(traj, t, global_cond=global_cond)
                # 3. compute previous image: x_t -> x_t-1
                traj = scheduler.step(
                    model_output, t, traj, 
                    generator=generator
                    ).prev_sample

        # finally make sure conditioning is enforced
        traj[condition_mask] = condition_data[condition_mask]        

        return traj
    
    def predict_action(self, inputs):

        obs, ee_pose, goal_position, spillage_classifier, start_guidance = inputs
        # obs, ee_pose = inputs

        # goal pose normalize
        # goal_position = _normalize(goal_position, self.input_max, self.input_min, self.input_mean)

        # condition through global feature
        # print(obs.shape, ee_pose.shape)
      
        obs_in = obs[:,:self.n_obs_steps,...]
        ee_in = ee_pose[:,:self.n_obs_steps,...]
        obs_features = self.obs_encoder((obs_in, ee_in))
        # # obs_features = self.obs_encoder(obs_in.reshape(-1,*obs.shape[2:]),
        # #                                 ee_in.reshape(-1, *ee_pose.shape[2:]))
        # global_cond = obs_features.reshape(obs.shape[0], -1)

        # obs_features = self.obs_encoder(obs_in.reshape(-1,*obs.shape[2:]))
        global_cond = obs_features.reshape(obs.shape[0], -1)

        cond_data = torch.zeros(size=(obs.shape[0], self.horizon, self.action_dim), device=self.device, dtype=torch.float32)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            global_cond=global_cond,
            goal_position=goal_position,
            spillage_classifier=None,
            obs_in=None,
            start_guidance=start_guidance
            )
        
        # unnormalize prediction
        naction_pred = nsample[...,:self.action_dim]
        action_pred = _denormalize(naction_pred, self.input_max, self.input_min, self.input_mean)

        # get action
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        return action, action_pred

# ====================================================================================================================================
def get_translation_matrix(x, y, z):
    return torch.tensor([
    [1, 0, 0, x],
    [0, 1, 0, y],
    [0, 0, 1, z],
    [0, 0, 0, 1]
    ])

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
def from_ee_to_spoon(ee_traj):
    
    # T_spoon_to_center = get_translation_matrix(0, -0.1, 0.15) # 15cm
    T_spoon_to_center = get_translation_matrix(0.03, 0, 0.17)
    ee_traj = ee_traj.squeeze(0) # (H, 9)
    spoon_traj = torch.zeros_like(ee_traj) # (H, 9)

    for i in range(ee_traj.shape[1]):
        ee = ee_traj[i]
        ee_pose = torch.eye(4)
        ee_pose[0:3, 3] = ee[0:3]
        ee_pose[0:3, 0:3] = rotation_6d_to_matrix(ee[3:])
        spoon_pose = torch.mm(ee_pose, T_spoon_to_center)
        spoon_traj[i, 0:3] = spoon_pose[0:3, 3]
        spoon_traj[i, 3:] = matrix_to_rotation_6d(spoon_pose[0:3, 0:3])

    return spoon_traj.unsqueeze(0) # (B, H, 9)

def _normalize(data, input_max, input_min, input_mean):
    ranges = input_max - input_min
    data = data.squeeze(0)
    data_normalize = torch.zeros_like(data)
    if data.shape[-1]>3:
        for i in range(3):
            if ranges[i] < 1e-4:
                # If variance is small, shift to zero-mean without scaling
                data_normalize[:, i] = data[:, i] - input_mean[i]
            else:
                # Scale to [-1, 1] range
                data_normalize[:, i] = -1 + 2 * (data[:, i] - input_min[i]) / ranges[i]   
        data_normalize[:, 3:] = data[:, 3:]
    else:
        for i in range(3):
            if ranges[i] < 1e-4:
                # If variance is small, shift to zero-mean without scaling
                data_normalize[:, i] = data[:, i] - input_mean[i]
            else:
                # Scale to [-1, 1] range
                data_normalize[i] = -1 + 2 * (data[i] - input_min[i]) / ranges[i]
    return data_normalize.unsqueeze(0)

def _denormalize(data, input_max, input_min, input_mean):
    ranges = input_max - input_min
    data = data.squeeze(0)
    data_denormalize = torch.zeros_like(data) # [8, 9]
    for i in range(3): # data.shape[1]
        if ranges[i] < 1e-4:
            # If variance is small, shift to zero-mean without scaling
            data_denormalize[:, i] = data[:, i] + input_mean
        else:
            # deScale to [-1, 1]
            data_denormalize[:, i] = ((data[:, i] + 1)*ranges[i] / 2) + input_min[i]
    data_denormalize[:, 3:] = data[:, 3:]
    return data_denormalize.unsqueeze(0)   

class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        local_cond_encoder = None
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            global_cond=None, **kwargs):

        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        # encode local features
        h_local = list()
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(self,
        action_dim, obs_dim,
        # obs mask setup
        max_n_obs_steps=2, 
        fix_obs_steps=True, 
        # action mask
        action_visible=False
        ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape, seed=None):
        device = self.device
        B, T, D = shape
        assert D == (self.action_dim + self.obs_dim)

        # create all tensors on this device
        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)

        # generate dim mask
        dim_mask = torch.zeros(size=shape, 
            dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[...,:self.action_dim] = True
        is_obs_dim = ~is_action_dim

        # generate obs mask
        if self.fix_obs_steps:
            obs_steps = torch.full((B,), 
            fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1, high=self.max_n_obs_steps+1, 
                size=(B,), generator=rng, device=device)
            
        steps = torch.arange(0, T, device=device).reshape(1,T).expand(B,T)
        obs_mask = (steps.T < obs_steps).T.reshape(B,T,1).expand(B,T,D)
        obs_mask = obs_mask & is_obs_dim

        # generate action mask
        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1, 
                torch.tensor(0,
                    dtype=obs_steps.dtype, 
                    device=obs_steps.device))
            action_mask = (steps.T < action_steps).T.reshape(B,T,1).expand(B,T,D)
            action_mask = action_mask & is_action_dim

        mask = obs_mask
        if self.action_visible:
            mask = mask | action_mask
        
        return mask

# ====================================================================================================================================    
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=0.75,
        min_value=0.0,
        max_value=0.9999
    ):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """

        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        self.decay = self.get_decay(self.optimization_step)

        all_dataptrs = set()
        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):            
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                # iterative over immediate parameters only.
                if isinstance(param, dict):
                    raise RuntimeError('Dict parameter not supported')
                
                # data_ptr = param.data_ptr()
                # if data_ptr != 0:
                #     all_dataptrs.add(data_ptr)

                if isinstance(module, _BatchNorm):
                    # skip batchnorms
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                elif not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)

        # verify that iterating over module and then parameters is identical to parameters recursively.
        # assert old_all_dataptrs == all_dataptrs
        self.optimization_step += 1

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.
    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """

    def __init__(
            self,
            input_shape=(4, 240, 320),
            num_kp=None,
            temperature=1.,
            learnable_temperature=False,
            output_variance=False,
            noise_std=0.0,
    ):
        """
        Args:
            input_shape (list, tuple): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not use spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape  # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(
                torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(
                torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self._in_w),
            np.linspace(-1., 1., self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(
            1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(
            1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
        self.fc = nn.Linear(self._in_c * 2, self._in_c)

        self.kps = None

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(num_kp={}, temperature={}, noise={})'.format(
            self._num_kp, self.temperature.item(), self.noise_std)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self._in_c)
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial
        probability distribution is created using a softmax, where the support is the
        pixel locations. This distribution is used to compute the expected value of
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.
        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert(feature.shape[1] == self._in_c)
        assert(feature.shape[2] == self._in_h)
        assert(feature.shape[3] == self._in_w)
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(
                self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(
                self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(
                self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat(
                [var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(),
                        feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()

        flat_features = feature_keypoints.view(feature_keypoints.shape[0], self._in_c * 2)
        outputs = self.fc(flat_features)
        return outputs