import torch
import torch.nn as nn
import numpy as np
import open3d as o3d

from models.base_models.encoders import (
    ee_pose_Encoder,
    property_Encoder,
    pcd_Encoder,
    depth_Encoder
)


class SensorFusion(nn.Module):
    """
    Regular SensorFusionNetwork Architecture
    Number of parameters:
    Inputs:
        pcd_info:      batch_size x 3 (file_num, pcd_env_num, pcd_index)
        top_pcd:       batch_size x 2000 x 3
        hand_pcd:      batch_size x 2000 x 3
        pcd_index:     batch_size x 1 
        pose_in:       batch_size x 1 x 7
        property_in:   batch_size x 1 x 4
        next_pose_in:  batch_size x 1 x 7
    """

    def __init__(
        self, device , z_dim=128, action_dim=4, encoder=False, deterministic=False, training_type = "spillage"
    ):
        super().__init__()

        self.z_dim = z_dim
        self.encoder_bool = encoder
        self.device = device
        self.deterministic = deterministic
        # self.feature_num = 3

        # Modality Encoders
        self.pcd_encoder = pcd_Encoder()
        self.depth_encoder = depth_Encoder()
        self.eepose_encoder = ee_pose_Encoder()


    def ee_normalize(self, data):
        input_range = torch.load('input_range.pt')
        self.input_max = input_range[0,:]
        self.input_min = input_range[1,:]
        self.input_mean = input_range[2,:]
  
        ranges = self.input_max - self.input_min
        data = data.squeeze(0)
        data_normalize = torch.zeros_like(data)
        for i in range(3):
            if ranges[i] < 1e-4:
                # If variance is small, shift to zero-mean without scaling
                data_normalize[:, i] = data[:, i] - self.input_mean[i]
            else:
                # Scale to [-1, 1] range
                data_normalize[:, i] = -1 + 2 * (data[:, i] - self.input_min[i]) / ranges[i]    
        data_normalize[:, 3:] = data[:, 3:]
        a,b,c = data_normalize.shape
        output = data_normalize.reshape(a, b*c)
        return output

    def forward_encoder(self, ee_pose, ee_pcd, hand_depth, front_depth, hand_pcd, front_pcd, hand_seg, front_seg):
      

        batch_size,image_num, _, _ = hand_depth.shape
  
        combine_depth = []
        combine_pcd = []
        combine_seg = []
        ee_pcd_list = []
   
        # Get encoded outputs
        # pose_out = self.ee_normalize(ee_pose)
        pose_out = self.eepose_encoder(ee_pose)             # shpae : torch.Size([batch_size , 7, 128])
       
        
        future_steps = ee_pose.shape[1]
        for i in range(future_steps):
            ee_pcd_out = self.pcd_encoder.encode(ee_pcd[:, i, :, :])
            ee_pcd_list.append(ee_pcd_out)

        all_ee_pcd = torch.cat(ee_pcd_list, dim=1)
     
        # hand_depth_out = self.depth_encoder(hand_depth)
        # front_depth_out = self.depth_encoder(front_depth)   # shpae : torch.Size([batch_size , 4, 128])
        # front_pcd = self.pcd_encoder.encode(front_pcd)
        # hand_pcd = self.pcd_encoder.encode(hand_pcd)

        for i in range(image_num):
            
            hand_depth_out = self.depth_encoder(hand_depth[:, i, :, :].unsqueeze(1))     # shpae : torch.Size([batch_size , 512])
            front_depth_out = self.depth_encoder(front_depth[:, i, :, :].unsqueeze(1))   # shpae : torch.Size([batch_size , 512])
            combine_depth.append(hand_depth_out)
            combine_depth.append(front_depth_out)


            # hand_pcd_out = self.pcd_encoder.encode(hand_pcd[:, i, :, :])     # shpae : torch.Size([batch_size , 256)
            front_pcd_out = self.pcd_encoder.encode(front_pcd[:, i, :, :])   # shpae : torch.Size([batch_size , 256])
      
            # combine_pcd.append(hand_pcd_out)
            combine_pcd.append(front_pcd_out)

            # hand_seg_out = hand_seg[:, i, :, :]    # shpae : torch.Size([batch_size , 256)
            front_seg_out = front_seg[:, i, :, :]  # shpae : torch.Size([batch_size , 256])
            # combine_seg.append(hand_seg_out)
            combine_seg.append(front_seg_out)
        
        all_depth = torch.cat(combine_depth, dim=0)
        all_pcd = torch.cat(combine_pcd, dim=1)
        all_seg = torch.cat(combine_seg, dim = 0)
    

        a, b = all_pcd.shape
        all_pcd = all_pcd.reshape(batch_size, int(a*b/batch_size))

        # all_depth = all_depth.reshape(batch_size, int(a*b/batch_size))

        # embeddings = torch.cat((pose_out, all_pcd, all_depth), 1).to(torch.float32)

        # embeddings = torch.cat((pose_out, all_pcd), 1).to(torch.float32)
       

        embeddings = torch.cat((all_pcd, all_ee_pcd), 1).to(torch.float32)

        # embeddings = combine_tensor.to(torch.float32)
        # embeddings = torch.cat((pose_out, hand_depth_out, front_depth_out), 1).to(torch.float32)

        # embeddings = torch.cat((pose_out, depth_out), 1).to(torch.float32)
        # embeddings = pose_out.to(torch.float32)

        return embeddings
    
    



class Dynamics_model(SensorFusion):
    """
    Regular SensorFusionNetwork Architecture
    Inputs:
    """

    def __init__(
        self, device,z_dim=128, action_dim=7, encoder=False, deterministic=False, training_type = "spillage"
    ):
        
        super().__init__(device, z_dim, action_dim, encoder, deterministic, training_type)
        self.multi_encoder = SensorFusion(device=device)

        if training_type == "spillage" or  training_type == "test": 
            num_type = 5
        else : 
            num_type = 6

        self.fc1 = nn.Linear(2560, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.0)

        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.0)

        # self.fc3 = nn.Linear(256, num_type)  # 6 output classes
        self.fc3 = nn.Linear(256, 3)
        
    def forward(
        self,
        ee_pose,
        ee_pcd, 
        hand_depth,
        front_depth,
        hand_pcd, 
        front_pcd,
        hand_seg,
        front_seg,
    ):

        # letent_z = self.multi_encoder.forward_encoder(hand_depth, hand_seg, ee_pose_in, front_depth, hand_pcd, front_pcd)
        letent_z = self.multi_encoder.forward_encoder(ee_pose, ee_pcd, hand_depth, front_depth, hand_pcd, front_pcd, hand_seg, front_seg)
        # letent_z = self.multi_encoder.forward_encoder(hand_depth, ee_pose_in, front_depth)
        x = letent_z.view(letent_z.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
 
        return x