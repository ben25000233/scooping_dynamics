import torch.nn as nn
from models.models_utils import init_weights
import numpy as np
import torch
from pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
class pcd_Encoder(nn.Module):
    def __init__(self, initailize_weights=True):
        super().__init__()
        hparams = {
            "model.use_xyz": True,
            "feature_num" : 0
        }
        self.model = PointNet2SemSegSSG(hparams)
        if initailize_weights:
            init_weights(self.modules())

        self.model = self.model.to(device)

        
    def encode(self, pcd):
       
        pcd_list = np.array(pcd)

        centroids = np.mean(pcd_list, axis=1, keepdims=True)
        # Normalize each point cloud by subtracting centroid and dividing by max distance
        pcd_list -= centroids
        max_distances = np.max(np.sqrt(np.sum(pcd_list**2, axis=2)), axis=1, keepdims=True)
        pcd_list /= max_distances[..., np.newaxis]

        # Convert to torch tensors
        points_tensor = torch.from_numpy(pcd_list).float()
        

        out_points_tensor = points_tensor.to(device)
        output_pcd = self.model(out_points_tensor)
        
        return output_pcd

class ee_pose_Encoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True):
      
        super().__init__()
        self.z_dim = z_dim

        self.eepose_encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 2 * self.z_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, eepose):
 
        # Ensure input dtype matches model parameters dtype
        eepose = eepose.to(next(self.eepose_encoder.parameters()).dtype)
        return self.eepose_encoder(eepose).unsqueeze(2)
    
class property_Encoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True):
      
        super().__init__()

        self.z_dim = z_dim

        self.property_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 2 * self.z_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, property):

        # Ensure input dtype matches model parameters dtype
        property = property.to(next(self.property_encoder.parameters()).dtype)
        
        return self.property_encoder(property).unsqueeze(2)
        
