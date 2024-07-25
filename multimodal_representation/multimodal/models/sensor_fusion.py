import torch
import torch.nn as nn
import open3d as o3d
import numpy as np
from models.models_utils import (
    duplicate,
    gaussian_parameters,
    product_of_experts,
    sample_gaussian,
    filter_depth,
)
from models.base_models.encoders import (
    ee_pose_Encoder,
    property_Encoder,
    pcd_Encoder,
)


class SensorFusion(nn.Module):
    """
    Regular SensorFusionNetwork Architecture
    Number of parameters:
    Inputs:
        pcd_info:      batch_size x 3 (file_num, pcd_env_num, pcd_index)
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
        self.feature_num = 4

        # Modality Encoders
        self.eepose_encoder = ee_pose_Encoder(self.z_dim)
        self.property_encoder = property_Encoder(self.z_dim)
        self.pcd_encoder = pcd_Encoder()


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
        # exit()
                
    def align_point_cloud(self, pcd, target_points=2000):

        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
        
        remaining_cloud = pcd.select_by_index(inliers, invert=True)
        num_points = len(remaining_cloud.points)
        
        if num_points > target_points:
            # Randomly downsample to target_points
            indices = np.random.choice(num_points, target_points, replace=False)
            new_pcd = remaining_cloud.select_by_index(indices)

        elif num_points < target_points:
            # Upsample to target_points
            additional_points_needed = target_points - num_points
            existing_points = np.asarray(remaining_cloud.points)
            
            # Repeat points to get the required number of points
            additional_indices = np.random.choice(num_points, additional_points_needed, replace=True)
            additional_points = existing_points[additional_indices]
            
            # Combine original points with additional points
            new_points = np.vstack((existing_points, additional_points))
            new_pcd.points = o3d.utility.Vector3dVector(new_points)
    
        return new_pcd

    def forward_encoder(self, pcd_info, pose_in, property_in, next_pose_in):
        # Batch size
        batch_size = pose_in.size()[0]
        top_pcd_list = []
        hand_pcd_list = []
   

        # Get encoded outputs
        pose_out = self.eepose_encoder(pose_in)
        next_pose_out = self.eepose_encoder(next_pose_in)
        property_out = self.property_encoder(property_in)
  
        
        #encode pcd
        for i in range(batch_size):
            file_num, pcd_env_num, pcd_index = pcd_info[0][i], pcd_info[1][i], pcd_info[2][i]
            
            top_pcd = o3d.io.read_point_cloud(f"./../../temp_collected_data/pcd_dataset/time_{file_num}/top_view/point_cloud/env_{pcd_env_num}_round_{pcd_index}.ply")
            
            top_allin_pcd = self.align_point_cloud(top_pcd)
      
            #hand_pcd = o3d.io.read_point_cloud(f"./../../collected_data/pcd_dataset/time_{file_num}/hand_view/point_cloud/env_{pcd_env_num}_round_{pcd_index}.ply")
            # o3d.visualization.draw_geometries([allin_pcd])
            top_pcd_list.append(top_allin_pcd.points)
       
        out_top_pcd_list = self.pcd_encoder.encode(top_pcd_list)
 
        #out_hand_pcd_list = self.pcd_encoder.encode(hand_pcd_list)


        m,n,k = out_top_pcd_list.shape
        out_top_pcd_list = out_top_pcd_list.reshape(m, n*k, 1).to(self.device)
        #out_hand_pcd_list = out_hand_pcd_list.reshape(m, n*k, 1).to(self.device)
        
        # Multimodal embedding
        mm_f1 = torch.cat([out_top_pcd_list, pose_out, next_pose_out, property_out], 1).squeeze()
        #mm_f1 = torch.cat([out_top_pcd_list, out_hand_pcd_list, pose_out, next_pose_out, property_out], 1).squeeze()
        mm_f2 = self.fusion_fc1(mm_f1)
        lentent_z = self.fusion_fc2(mm_f2)

        return lentent_z


    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]


class Dynamics_MLP(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, output_size=1, dropout_prob=0.3):
        super(Dynamics_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class Dynamics_model(SensorFusion):
    """
    Regular SensorFusionNetwork Architecture
    Inputs:
    """

    def __init__(
        self, device,z_dim=128, action_dim=7, encoder=False, deterministic=False,
    ):
        super().__init__(device, z_dim, action_dim, encoder, deterministic)
        self.deterministic = deterministic

        self.dynamic_model = Dynamics_MLP(z_dim, z_dim, 1)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(
        self,
        pcd_info,
        pose_in,
        property_in,
        next_pose_in
    ):

        letent_z = self.forward_encoder(pcd_info, pose_in, property_in, next_pose_in)
       
        # Training Objectives
        pred_spillage = self.dynamic_model(letent_z)

        return pred_spillage
    
