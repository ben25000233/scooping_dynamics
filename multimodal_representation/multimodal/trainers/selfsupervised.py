from __future__ import print_function
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import copy

from models.sensor_fusion import Dynamics_model
from utils import (
    compute_accuracy,
    set_seeds,
)

from dataloaders import MultimodalManipulationDataset
from torch.utils.data import DataLoader

class selfsupervised:
    def __init__(self, configs):

        # ------------------------
        # Sets seed and cuda
        # ------------------------
        use_cuda = configs["cuda"] and torch.cuda.is_available()

        self.configs = configs
        self.device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:
            print("Let's use", torch.cuda.device_count(), "GPUs!")


        set_seeds(configs["seed"], use_cuda)

        # model
        self.model = Dynamics_model(
            device=self.device,
            encoder=configs["encoder"],
            deterministic=configs["deterministic"],
            z_dim=configs["zdim"],
            action_dim=configs["action_dim"],
        ).to(self.device)



        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.configs["lr"],
            betas=(self.configs["beta1"], 0.999),
            weight_decay=0.0,
        )

        self.loss_function = nn.BCEWithLogitsLoss()

        # Weights for input
        self.alpha_top_pcd = configs["top_pcd"]
        self.alpha_hand_pcd = configs["hand_pcd"]
        self.alpha_property_in = configs["property_in"]
        self.alpha_eepose = configs["eepose"]


        # ------------------------
        # Handles Initialization
        # ------------------------
        if configs["load"]:
            self.load_model(configs["model_path"])

        self._init_dataloaders()

        if not os.path.exists("./../../ckpt"):
            os.makedirs("./../../ckpt")
 

    def train(self):

        best_loss = 0   
        best_model_wts = None     

        for i_epoch in range(self.configs["max_epoch"]):
            # ---------------------------
            # Train Step
            # ---------------------------
            print(f"epoch {i_epoch}")
            total_loss = 0.0
            total_acc = 0.0
            self.model.train()
            for idx, sample_batched in enumerate(self.dataloaders["train"]):
                self.optimizer.zero_grad()
                loss, acc = self.loss_calc(sample_batched)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_acc += acc.item()

            

            print(f"train_loss : {total_loss/self.len_data} train_accuracy : {total_acc/self.len_data}")
            val_loss= self.validate()
            

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts=copy.deepcopy(self.model.state_dict())
            
            if (i_epoch+1) % 10 == 0 :
                FILE = f"./../../ckpt/epoch{i_epoch}.pt"
                torch.save(best_model_wts,FILE)


    def validate(self):

        total_loss = 0.0
        total_acc = 0.0
        self.model.eval()

        for i_iter, sample_batched in enumerate(self.dataloaders["val"]):
            
            loss, acc = self.loss_calc(sample_batched)
            total_loss += loss.item()
            total_acc += acc.item()

        print(f"val_loss : {total_loss/self.val_len_data} val_accuracy : {total_acc/self.val_len_data}")
        return total_loss/self.val_len_data

    def load_model(self, path):
        print("Loading model from {}...".format(path))
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt)

    def loss_calc(self, sampled_batched):

        pcd_info = sampled_batched["pcd_info"]
        property = sampled_batched["property"].to(self.device)
        eepose = sampled_batched["eepose"].to(self.device)
        next_eepose = sampled_batched["next_eepose"].to(self.device)
        #label
        spillage = sampled_batched["spillage_amount"].to(self.device).float()
        
        spillage = spillage.reshape(spillage.shape[0], 1)
        binary_spillage = spillage.clone()

        # trans for binary label
        for i in range(len(spillage)):
            if spillage[i] > 0:
                binary_spillage[i] = 1
            else : 
                binary_spillage[i] = 0
     

        pred_spillage = self.model(
                pcd_info, eepose, property, next_eepose
            )
        
        loss = self.loss_function(pred_spillage, binary_spillage)
    
        sig_pred = nn.Sigmoid()(pred_spillage).detach()
        spillage_accuracy = compute_accuracy(sig_pred, spillage.detach())

        # error checking
        '''
        import open3d as o3d
        for i in range(len(pred_spillage)):
            if pred_spillage[i] > 0.5:
                pred = 1
            else:
                pred = 0
            if pred!= binary_spillage[i]:
                print(f"gd : {spillage[i]}  pred : {pred}")
                print(pcd_info[0][i],pcd_info[1][i],pcd_info[2][i])
                pcd = o3d.io.read_point_cloud(f"./../../collected_data/pcd_dataset/time_{pcd_info[0][i]}/hand_view/point_cloud/env_{pcd_info[1][i]}_round_{pcd_info[2][i]}.ply")
                o3d.visualization.draw_geometries([pcd])
        '''
        
        
        return (
            loss,
            spillage_accuracy,
        )

    def _init_dataloaders(self):

        filename_list = []
        for file in os.listdir(self.configs["dataset"]):
            if file.endswith(".h5"):
                filename_list.append(self.configs["dataset"] + file)
  
        print(
            "Number of files in multifile dataset = {}".format(len(filename_list))
        )

        val_filename_list = []

        val_index = np.random.randint(
            0, len(filename_list), int(len(filename_list) * self.configs["val_ratio"])
        )
        

        for index in val_index:
            val_filename_list.append(filename_list[index])


        # move all val files from filename list
        while val_index.size > 0:
            filename_list.pop(val_index[0])
            val_index = np.where(val_index > val_index[0], val_index - 1, val_index)
            val_index = val_index[1:]

       
        print("Initial finished")

        self.dataloaders = {}
        self.samplers = {}
        self.datasets = {}


        self.datasets["train"] = MultimodalManipulationDataset(
            filename_list,
            data_length = self.configs["num_envs"] * self.configs["collect_time"] * self.configs["n_time_steps"],
            training_type=self.configs["training_type"],
            action_dim=self.configs["action_dim"],
            single_env_steps = self.configs["collect_time"] * self.configs["n_time_steps"],
            type = "training",
        )

        self.datasets["val"] = MultimodalManipulationDataset(
            val_filename_list,
            data_length = self.configs["num_envs"] * self.configs["collect_time"] * self.configs["n_time_steps"],
            training_type=self.configs["training_type"],
            action_dim=self.configs["action_dim"],
            single_env_steps = self.configs["collect_time"] * self.configs["n_time_steps"],
            type = "validation",
        )

        print("Dataset finished")

        self.dataloaders["val"] = DataLoader(
            self.datasets["val"],
            batch_size=self.configs["batch_size"],
            num_workers=self.configs["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

        self.dataloaders["train"] = DataLoader(
            self.datasets["train"],
            batch_size=self.configs["batch_size"],
            num_workers=self.configs["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

        self.len_data = len(self.dataloaders["train"])
        self.val_len_data = len(self.dataloaders["val"])
        

        print("Finished setting up date")

    