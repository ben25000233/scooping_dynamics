from pyconfigparser import configparser
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from argparse import ArgumentParser
import cv2
from typing import Union
import pytorch3d.transforms as pt
import functools
# from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d


# diffusion_policy/diffusion_policy/model/common/rotation_transformer.py
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
    
# diffusion_policy/diffusion_policy/model/common/normalizer.py
def _normalize(data, input_max, input_min, input_mean):
    ranges = input_max - input_min
    data_normalize = np.zeros_like(data)
    for i in range(3):
        if ranges[i] < 1e-4:
            # If variance is small, shift to zero-mean without scaling
            data_normalize[:, i] = data[:, i] - input_mean[i]
        else:
            # Scale to [-1, 1] range
            data_normalize[:, i] = -1 + 2 * (data[:, i] - input_min[i]) / ranges[i]    
    data_normalize[:, 3:] = data[:, 3:]
    return data_normalize

# origin:[480, 640] -> resize:[240, 320] (h, w)
def rgb_transform(img):
    return transforms.Compose([
        transforms.Resize([240, 320]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])(img)

def depth_transform(img):
    return transforms.Compose([
        transforms.Resize([240, 320]),
        transforms.ToTensor(),
    ])(img)

def main():

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", default="grits.yaml")
    args = parser.parse_args()
    cfg = configparser.get_config(file_name=args.config)

    # ========================================= #
    # grits.yaml
    # ---------------------------------
    # T=16, To=5, Ta=8
    # |o|o|o|o|o|
    # | | | | |a|a|a|a|a|a|a|a|
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|
    # ========================================= #

    dir_path = "./sim_dataset"
    out_dir = "./process_dataset"
    food_item = cfg.data.food_item
    
    # food category
    num_all = 0
    rotation_transformer = RotationTransformer('quaternion', 'rotation_6d')
    
    ee_all = []
    for i in range(len(food_item)):
        for j in range(cfg.data.trial_num):
            ee_pose = np.load(dir_path + "/" + food_item[i] + "_{}/ee_pose_qua.npy".format(str(j+1)))
            ee_all.append(ee_pose)

    ee_all = np.asarray(ee_all)
    ee_all = np.reshape(ee_all, (ee_all.shape[0]*ee_all.shape[1],ee_all.shape[2]))
    input_max = np.max(ee_all, axis=0)
    input_min = np.min(ee_all, axis=0)
    input_mean = np.mean(ee_all, axis=0)
    input_range = torch.from_numpy(np.concatenate((
        np.expand_dims(input_max, 0), 
        np.expand_dims(input_min, 0), 
        np.expand_dims(input_mean, 0)), 0))
    torch.save(input_range, out_dir + "/input_range.pt")
    print("input_max= ", input_max)
    print("input_min= ", input_min)
    print("input_mean= ", input_mean)

    for i in range(len(food_item)):
        for j in range(cfg.data.trial_num):
            print(food_item[i] + '_' + str(j+1))

            rgb_front_orin = np.load(dir_path + "/" + food_item[i] + "_{}/rgb_front.npy".format(str(j+1)))
            depth_front_orin = np.load(dir_path + "/" + food_item[i] + "_{}/depth_front.npy".format(str(j+1)))
            ee_pose_orin = np.load(dir_path + "/" + food_item[i] + "_{}/ee_pose_qua.npy".format(str(j+1)))

            rgb_front = rgb_front_orin[10:260]
            depth_front = depth_front_orin[10:260]
            ee_pose = ee_pose_orin[10:260]

            assert rgb_front.shape[0]==250
            assert depth_front.shape[0]==250
            assert ee_pose.shape[0]==250

            # convert qua to rotation 6D
            ee_pose_position = ee_pose[:, :3]
            ee_pose_rotation = rotation_transformer.forward(ee_pose[:, 3:])
            ee_pose_6d = np.concatenate((ee_pose_position, ee_pose_rotation), -1) # [:,9]

            # add To-1 frame before first frame
            To = cfg.n_obs_steps
            for _ in range(To-1):
                rgb_front = np.concatenate((np.expand_dims(rgb_front[0], 0), rgb_front), 0)
                depth_front = np.concatenate((np.expand_dims(depth_front[0], 0), depth_front), 0)
                ee_pose_6d = np.concatenate((np.expand_dims(ee_pose_6d[0], 0), ee_pose_6d), 0)               

            assert rgb_front.shape[0]==250+To-1
            assert depth_front.shape[0]==250+To-1
            assert ee_pose_6d.shape[0]==250+To-1

            T = cfg.horizon
            t=0
            while t<ee_pose.shape[0]-T: # 254-16=238
                    
                rgb_front_tslice = rgb_front[t:t+T] # [16]
                depth_front_tslice = depth_front[t:t+T] # [16,]
                ee_pose_6d_tslice = ee_pose_6d[t:t+T] # [16, 7]
                # action normalize to [-1,1], and transfer to tensor
                ee_6d_tslice_normalize = _normalize(ee_pose_6d_tslice, input_max, input_min, input_mean)

                ob_front_tensor = []
                for tt in range(T):
                    # observation transform
                    # rgb
                    rgb_PIL =  Image.fromarray(rgb_front_tslice[tt].astype('uint8'), 'RGB')
                    rgb_front_tt = rgb_transform(rgb_PIL)
                    # depth
                    depth_PIL = depth_front_tslice[tt].astype(np.float32)
                    depth_PIL = depth_PIL / np.max(depth_PIL)
                    depth_front_tt = depth_transform(Image.fromarray(depth_PIL))
                    # concat
                    ob_front_tensor.append(torch.cat((rgb_front_tt, depth_front_tt), 0))            
                
                # save tensor
                ee_6d_tensor = torch.from_numpy(ee_6d_tslice_normalize) # (16, 9)
                ob_front_tensor = torch.stack(ob_front_tensor, dim=0)
                assert ob_front_tensor.shape[0]==T
                assert ee_6d_tensor.shape[0]==T
                torch.save(ob_front_tensor, out_dir+'/ob_front_{}.pt'.format(str(num_all).zfill(5)))
                torch.save(ee_6d_tensor, out_dir+'/traj_{}.pt'.format(str(num_all).zfill(5)))
                num_all+=1
                print(num_all) # one trial = 234 .pt file   

                t+=1 
          

if __name__ == '__main__':
    main()
