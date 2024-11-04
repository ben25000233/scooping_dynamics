import torch
import torch.nn as nn
import numpy as np
import open3d as o3d

from models.base_models.encoders import (
    ee_pose_Encoder,
    property_Encoder,
    obs_pcd_Encoder,
    flow_pcd_Encoder,
    depth_Encoder
)

def check_pcd_color(pcd):

    color_map = {
        0: [1, 0, 0],    # Red
        1: [0, 1, 0],    # Green
        2: [0, 0, 1],    # Blue
        3: [1, 1, 0],    # Yellow
        4: [1, 0, 1]     # Magenta
    }
    points = []
    colors = []
   
    
    for i in range(pcd.shape[0]):
        points.append(pcd[i][:3].cpu())
        colors.append(color_map[pcd[i][3].cpu().item()])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([point_cloud])

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
        self.obs_pcd_encoder = obs_pcd_Encoder()
        self.flow_pcd_encoder = flow_pcd_Encoder()
        # self.depth_encoder = depth_Encoder()
        # self.eepose_encoder = ee_pose_Encoder()

        self.pcd_info = np.load("pcd_nor_info.npy", allow_pickle=True)
      

    def nor_pcd(self, pcd_list, type = "tool"):
        if type == "tool":
            centroids = torch.tensor(self.pcd_info[0], dtype = torch.float32).to(self.device)
            max_distances = torch.tensor(self.pcd_info[1], dtype = torch.float32).to(self.device)
        else : 
            centroids = torch.tensor(self.pcd_info[2], dtype = torch.float32).to(self.device)
            max_distances = torch.tensor(self.pcd_info[3], dtype = torch.float32).to(self.device)

        xyz_pcd = pcd_list[..., :3]
        # Assuming pcd_list is a PyTorch tensor, calculate centroids for each point cloud
        # centroids = xyz_pcd.mean(dim=1, keepdim=True)
   
        # Subtract centroids from each point cloud
        nor_pcd = xyz_pcd - centroids

        # Calculate the max distance from the centroid for each point cloud
        # max_distances = torch.max(torch.sqrt(torch.sum(nor_pcd**2, dim=2)), dim=1, keepdim=True).values

        # Normalize by the max distances
        nor_pcd /= max_distances[..., None]

        # If there are additional features, append them back
        if pcd_list.shape[-1] > 3:
            nor_pcd = torch.cat((nor_pcd, pcd_list[..., 3:]), dim=2)

        return nor_pcd


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

    # def forward_encoder(self, ee_pose, ee_pcd, hand_depth, front_depth, hand_pcd, front_pcd, hand_seg, front_seg):
    def forward_encoder(self, ee_pose, front_pcd, ee_pcd, flow_pcd):
        print(ee_pose.shape)
        print(front_pcd.shape)
        exit()
        batch_size,image_num, _, _ = front_pcd.shape
  
        combine_depth = []
        combine_pcd = []
        combine_seg = []
        ee_pcd_list = []
        flow_pcd_list = []
   
        # Get encoded outputs

        # nor_pose = self.ee_normalize(ee_pose)
        # pose_out = self.eepose_encoder(nor_pose.reshape(ee_pose.shape))             # shpae : torch.Size([batch_size , 7, 128])
   
        
        
        future_steps = ee_pose.shape[1]
        '''
        for i in range(future_steps):
            
            check_pcd_color(ee_pcd[0, i, :, :])
            ee_pcd_out = self.obs_pcd_encoder.encode(ee_pcd[:, i, :, :])
            ee_pcd_list.append(ee_pcd_out)
        all_ee_pcd = torch.cat(ee_pcd_list, dim=1)
        '''
        
       
        
        # ee_pcd_out_1 = self.obs_pcd_encoder.encode(self.nor_pcd(ee_pcd[:, 0, :, :]))
        # ee_pcd_out_2 = self.obs_pcd_encoder.encode(self.nor_pcd(ee_pcd[:, -1, :, :]))
        # all_ee_pcd = torch.cat([ee_pcd_out_1, ee_pcd_out_2], dim=1)
        

        flow_pcd_out = self.flow_pcd_encoder.encode(flow_pcd.type(torch.float32))
        

        # hand_depth_out = self.depth_encoder(hand_depth)
        # front_depth_out = self.depth_encoder(front_depth)   # shpae : torch.Size([batch_size , 4, 128])
        # front_pcd = self.pcd_encoder.encode(front_pcd)
        # hand_pcd = self.pcd_encoder.encode(hand_pcd)

        for i in range(image_num):
            
            # hand_depth_out = self.depth_encoder(hand_depth[:, i, :, :].unsqueeze(1))     # shpae : torch.Size([batch_size , 512])
            # front_depth_out = self.depth_encoder(front_depth[:, i, :, :].unsqueeze(1))   # shpae : torch.Size([batch_size , 512])
            # combine_depth.append(hand_depth_out)
            # combine_depth.append(front_depth_out)


            # hand_pcd_out = self.pcd_encoder.encode(hand_pcd[:, i, :, :])     # shpae : torch.Size([batch_size , 256)
            
            # front_pcd_out = self.obs_pcd_encoder.encode(front_pcd[:, i, :, :])   # shpae : torch.Size([batch_size , 256]
            # check_pcd_color(front_pcd[0,0, :, :])
          

            front_pcd_out = self.obs_pcd_encoder.encode(self.nor_pcd(front_pcd[:, i, :, :]))   # shpae : torch.Size([batch_size , 256])
        
            # combine_pcd.append(hand_pcd_out)
            combine_pcd.append(front_pcd_out)

            # hand_seg_out = hand_seg[:, i, :, :]    # shpae : torch.Size([batch_size , 256)
            # front_seg_out = front_seg[:, i, :, :]  # shpae : torch.Size([batch_size , 256])
            # combine_seg.append(hand_seg_out)
            # combine_seg.append(front_seg_out)
        
        # all_depth = torch.cat(combine_depth, dim=0)
        all_pcd = torch.cat(combine_pcd, dim=1)

  
        a, b = all_pcd.shape
        all_pcd = all_pcd.reshape(batch_size, int(a*b/batch_size))

        # embeddings = torch.cat((all_ee_pcd, all_pcd), 1).to(torch.float32)

        # embeddings = torch.cat((pose_out, all_pcd), 1).to(torch.float32)

        # print(all_pcd.shape, flow_pcd_out.shape)
        embeddings = torch.cat((all_pcd, flow_pcd_out), 1).to(torch.float32)

        # embeddings = combine_tensor.to(torch.float32)
        # embeddings = torch.cat((pose_out, hand_depth_out, front_depth_out), 1).to(torch.float32)

        # embeddings = torch.cat((pose_out, depth_out), 1).to(torch.float32)
        # embeddings = pose_out.to(torch.float32)

        return embeddings





class Dynamics_model(SensorFusion):
    """
    LSTM-based SensorFusion Network Architecture
    Inputs:
    """

    def __init__(
        self, device, z_dim=128, action_dim=7, encoder=False, deterministic=False, training_type="spillage", lstm_hidden_size=256, num_layers=1
    ):
        super().__init__(device, z_dim, action_dim, encoder, deterministic, training_type)
        self.multi_encoder = SensorFusion(device=device)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=768, hidden_size=512, num_layers=num_layers, batch_first=True)


        self.fc1 = nn.Linear(512, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)

        self.fc2 = nn.Linear(256, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.1)

        self.fc3 = nn.Linear(32, 2)  # Output layer for 3 predictions

    def forward(self, ee_pose, tool_with_ball_pcd, ee_pcd, flow_pcd):

        # Get latent representation from multi-encoder
        latent_z = self.multi_encoder.forward_encoder(ee_pose, tool_with_ball_pcd, ee_pcd, flow_pcd)

        # Prepare the input for LSTM: add a sequence dimension
        latent_z = latent_z.unsqueeze(1)  # Shape: (batch_size, seq_len=1, z_dim)

        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(latent_z)

        # Use the last hidden state
        lstm_out = h_n[-1]  # Get the output of the last LSTM layer for the last time step

        # Fully connected layers
        x = self.fc1(lstm_out)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)

        return x
