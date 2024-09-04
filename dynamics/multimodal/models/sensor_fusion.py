import torch
import torch.nn as nn
import open3d as o3d
import numpy as np

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
        self, device , z_dim=128, action_dim=4, encoder=False, deterministic=False
    ):
        super().__init__()

        self.z_dim = z_dim
        self.encoder_bool = encoder
        self.device = device
        self.deterministic = deterministic
        self.feature_num = 3

        # Modality Encoders
        # self.eepose_encoder = ee_pose_Encoder(self.z_dim)
        # self.property_encoder = property_Encoder(self.z_dim)
        # self.pcd_encoder = pcd_Encoder()
        self.depth_encoder = depth_Encoder()
        self.eepose_encoser = ee_pose_Encoder()


        # Modality fusion network
        # 5 Total modalities each (2 * z_dim)
            
        self.fusion_fc1 = nn.Sequential(
            nn.Linear(self.feature_num * 2 * self.z_dim, 128), nn.LeakyReLU(0.1, inplace=True)
        )
        self.fusion_fc2 = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim), nn.LeakyReLU(0.1, inplace=True)
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        input_range = torch.load("./../input_range.pt")
        self.input_max = input_range[0,:]
        self.input_min = input_range[1,:]
        self.input_mean = input_range[2,:]

        # exit()
                
    def ee_normalize(self, data):
  
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

    def forward_encoder(self, hand_depth, hand_seg, ee_pose):

   
        # Get encoded outputs
        # pose_out = self.ee_normalize(ee_pose)
        
        pose_out = self.eepose_encoser(ee_pose)
        depth_out = self.depth_encoder(hand_depth)
        seg_out = self.depth_encoder(hand_seg)


        # embeddings = torch.cat((pose_out, depth_out, seg_out), 1).to(torch.float32)
        embeddings = torch.cat((pose_out, depth_out), 1).to(torch.float32)
        # embeddings = pose_out.to(torch.float32)
  

        return embeddings



class Dynamics_model(SensorFusion):
    """
    Regular SensorFusionNetwork Architecture
    Inputs:
    """

    def __init__(
        self, device,z_dim=128, action_dim=7, encoder=False, deterministic=False,
    ):
        
        super().__init__(device, z_dim, action_dim, encoder, deterministic)
        self.multi_encoder = SensorFusion(device=device)

        self.fc1 = nn.Linear(16 * 128, 512)  # Flatten the input
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(256, 6)  # 6 output classes
        
        
    def forward(
        self,
        hand_depth,
        hand_seg,
        ee_pose_in,
    ):

        letent_z = self.multi_encoder.forward_encoder(hand_depth, hand_seg, ee_pose_in)
        x = letent_z.view(letent_z.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
 
        return x
        
