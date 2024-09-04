import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from models.sensor_fusion import Dynamics_model

from dataloaders import MultimodalManipulationDataset
from torch.utils.data import DataLoader
from copy import copy

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
       

        # model
        self.model = Dynamics_model(
            device=self.device,
            encoder=configs["encoder"],
            deterministic=configs["deterministic"],
            z_dim=configs["zdim"],
            action_dim=configs["action_dim"],
        ).to(self.device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)


        self.loss_function = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.configs["lr"],
            betas=(self.configs["beta1"], 0.999),
            weight_decay=0.0,
        )

        

        # Weights for input
        self.alpha_vision = configs["vision"]
        self.alpha_depth = configs["depth"]
        self.alpha_eepose = configs["eepose"]


        # ------------------------
        # Handles Initialization
        # ------------------------
        if configs["load"]:
            self.load_model(configs["model_path"])

        self._init_dataloaders()

        if not os.path.exists("./../ckpt"):
            os.makedirs("./../ckpt")
 

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
            train_loader = self.dataloaders["train"]
   
            for idx, sample_batched in enumerate(tqdm(train_loader)):

                self.optimizer.zero_grad()
                loss, acc = self.loss_calc(sample_batched)
 
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_acc += acc

            print(f"train_loss : {total_loss/self.len_data} train_accuracy : {total_acc/self.len_data}")
            # del train_loader
            # gc.collect()

            val_loss= self.validate()
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts=copy.deepcopy(self.model.state_dict())
            
            if (i_epoch+1) % 10 == 0 :
                FILE = f"./../ckpt/epoch{i_epoch}.pt"
                torch.save(best_model_wts,FILE)


    def validate(self):

        total_loss = 0.0
        total_acc = 0.0
        self.model.eval()
        val_loader = self.dataloaders["val"]

        for i_iter, sample_batched in enumerate(tqdm(val_loader)):
            
            loss, acc = self.loss_calc(sample_batched)
            total_loss += loss.item()
            total_acc += acc
        
        print(f"val_loss : {total_loss/self.val_len_data} val_accuracy : {total_acc/self.val_len_data}")
        return total_loss/self.val_len_data

    def load_model(self, path):
        print("Loading model from {}...".format(path))
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt)

    def loss_calc(self, sampled_batched):

        seg = sampled_batched["hand_seg"].to(self.device)
        depth = sampled_batched["hand_depth"].to(self.device)
        eepose = sampled_batched["eepose"].to(self.device)
        #label
        spillage = sampled_batched["spillage_vol"].to(self.device)
        
        # spillage = spillage.reshape(spillage.shape[0], 1).to(torch.float32)


        pred_spillage = self.model(
                depth,seg, eepose, 
            )
      
        
        loss = self.loss_function(pred_spillage, spillage)

        acc_num = 0

        for idx, pre_spillage in enumerate(pred_spillage):
            gd_class = torch.argmax(spillage[idx]).item()
            pre_class = torch.argmax(pre_spillage).item()
            if gd_class == pre_class :            
                acc_num += 1
    
        return (
            loss,
            acc_num/len(pred_spillage),
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

    