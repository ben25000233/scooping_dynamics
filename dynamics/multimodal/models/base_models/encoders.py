import torch.nn as nn
import numpy as np
import torch
from pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG
from models.base_models.layers import CausalConv1D, Flatten, conv2d
import torchvision.models as models


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
    def __init__(self, proprio_dim=7):
        super(ee_pose_Encoder, self).__init__()
        
        self.eepose_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )

    def forward(self, eepose):
  
        # Ensure input dtype matches model parameters dtype
        eepose = eepose.to(next(self.eepose_encoder.parameters()).dtype)
        return self.eepose_encoder(eepose)
    
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
        
class ImageEncoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True):
        """
        Image encoder taken from Making Sense of Vision and Touch
        """
        super().__init__()
        self.z_dim = z_dim

        self.img_conv1 = conv2d(3, 16, kernel_size=7, stride=2)
        self.img_conv2 = conv2d(16, 32, kernel_size=5, stride=2)
        self.img_conv3 = conv2d(32, 64, kernel_size=5, stride=2)
        self.img_conv4 = conv2d(64, 64, stride=2)
        self.img_conv5 = conv2d(64, 128, stride=2)
        self.img_conv6 = conv2d(128, self.z_dim, stride=2)
        self.img_encoder = nn.Linear(4 * self.z_dim, 2 * self.z_dim)
        self.flatten = Flatten()

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, image):
        # image encoding layers
        out_img_conv1 = self.img_conv1(image)
        out_img_conv2 = self.img_conv2(out_img_conv1)
        out_img_conv3 = self.img_conv3(out_img_conv2)
        out_img_conv4 = self.img_conv4(out_img_conv3)
        out_img_conv5 = self.img_conv5(out_img_conv4)
        out_img_conv6 = self.img_conv6(out_img_conv5)

        img_out_convs = (
            out_img_conv1,
            out_img_conv2,
            out_img_conv3,
            out_img_conv4,
            out_img_conv5,
            out_img_conv6,
        )

        # image embedding parameters
        flattened = self.flatten(out_img_conv6)
        img_out = self.img_encoder(flattened).unsqueeze(2)

        return img_out, img_out_convs

class depth_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        num_classes = 6
        self.hand_depth_encoder = models.resnet18(pretrained=False)
        self.hand_depth_encoder.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.hand_depth_encoder.fc = nn.Linear(self.hand_depth_encoder.fc.in_features, num_classes)
        self.hand_depth_encoder.fc = nn.Identity()
   

    def forward(self, depth):
        # depth encoding layers
        depth_out = self.hand_depth_encoder(depth)
        batch_size, chennal = depth_out.shape
        depth_out = depth_out.reshape(batch_size,  int(chennal / 128) , 128)
        return depth_out
    

