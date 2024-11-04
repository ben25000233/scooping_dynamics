import pytorch_lightning as pl
import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from torch.utils.data import DataLoader

from pointnet2.data import Indoor3DSemSeg
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG


class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[self.hparams["feature_num"], 32, 32, 64],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )

        # self.FP_modules = nn.ModuleList()
        # self.FP_modules.append(PointnetFPModule(mlp=[128 , 128, 128, 128]))
        # self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        # self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        # self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            # nn.Dropout(0.1),
            nn.Conv1d(256, 16, kernel_size=1),
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
   
        xyz, features = self._break_up_pc(pointcloud)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        
        #return features
        return self.fc_layer(features)
        
        l_xyz, l_features = [xyz], [features]
    
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
       
        # for i in range(-1, -(len(self.FP_modules) + 1), -1):
        #     l_features[i - 1] = self.FP_modules[i](
        #         l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
        #     )
        #return self.fc_lyaer(l_features[0])
        return l_features[-1]

