from __future__ import print_function
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from models.sensor_fusion import Dynamics_model

from dataloaders import MultimodalManipulationDataset
from torch.utils.data import DataLoader
import copy
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class predict_spillage:
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
            training_type = "spillage",
        ).to(self.device)

        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)


        # self.loss_function = nn.CrossEntropyLoss()
        self.loss_function = nn.BCEWithLogitsLoss()


        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.configs["lr"],
            betas=(self.configs["beta1"], 0.999),
            weight_decay=0.0,
        )
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.configs["lr"], momentum=0.99, weight_decay=0.02)

        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.9)

        

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

        if not os.path.exists("./../../spillage_ckpt"):
            os.makedirs("./../../spillage_ckpt")
 

    def train(self):
        train_loader = self.dataloaders["train"]
        val_loader = self.dataloaders["val"]
        best_loss = float("inf") 
        best_model_wts = None    
        
        # TensorBoard writer
        writer = SummaryWriter(log_dir='./logs')

        for i_epoch in range(self.configs["max_epoch"]):
            self.model.train() 
            print(f"epoch {i_epoch}")
            total_loss = 0.0
            total_train_acc = 0.0

            self.gd_list = [0, 0]
            self.acc_list = [0, 0]

            for idx, sample_batched in enumerate(tqdm(train_loader)):

                self.optimizer.zero_grad()
                loss, acc, _, _ = self.loss_calc(sample_batched)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_train_acc += acc
            
            self.scheduler.step()

            train_loss = total_loss / len(train_loader)
            train_acc = total_train_acc / len(train_loader)
            print(f"train_loss: {train_loss} train_accuracy: {train_acc}")
            print(f"total_num : {self.gd_list}")
            print(f"acc_num   : {self.acc_list}")
            for i in range(len(self.gd_list)):
                print(f"level_acc_{i} : {self.acc_list[i]/self.gd_list[i]}")


            # Log training loss and accuracy to TensorBoard
            writer.add_scalar('Loss/train', train_loss, i_epoch)
            writer.add_scalar('Accuracy/train', train_acc, i_epoch)

            # Validation
            self.gd_list = [0, 0]
            self.acc_list = [0, 0]

            val_loss = self.validate(val_loader, writer, i_epoch)
            print(f"total_num : {self.gd_list}")
            print(f"acc_num   : {self.acc_list}")
            for i in range(len(self.gd_list)):
                print(f"level_acc_{i} : {self.acc_list[i]/self.gd_list[i]}")
            
            
            # Save the best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
            
            if (i_epoch + 1) % 50 == 0:
                FILE = f"./../../spillage_ckpt/epoch{i_epoch}.pt"
                torch.save(best_model_wts, FILE)

        writer.close()

    def validate(self, val_loader, writer, epoch):
        total_loss = 0.0
        total_acc = 0.0
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i_iter, sample_batched in enumerate(tqdm(val_loader)):
                loss, acc, pred, label = self.loss_calc(sample_batched)
                _, preds = torch.max(pred, 1)
                _, labels = torch.max(label, 1)


                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                
                total_loss += loss.item()
                total_acc += acc

        if (epoch+1) % 5 == 0 :
            self.show_confusion(all_labels, all_preds, epoch)
        
        val_loss = total_loss / len(val_loader)
        val_acc = total_acc / len(val_loader)
        
        # Log validation loss and accuracy to TensorBoard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f"val_loss: {val_loss} val_accuracy: {val_acc}")
        return val_loss
    
    def show_confusion(self, label, prediction, epoch):
        class_labels = np.argmax(label, axis=1)
        
        cm = confusion_matrix(class_labels, prediction)
    
        # Plot confusion matrix using seaborn
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        save_path = f"./confusion/epoch_{epoch}.png"
        plt.savefig(save_path, format='png')
        # plt.show()
    
    def load_model(self, path):
        print("Loading model from {}...".format(path))
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt)

    def loss_calc(self, sampled_batched, run_model_type = "train"):
        
        eepose = sampled_batched["eepose"].to(self.device)
        ee_pcd = sampled_batched["ee_pcd"].to(self.device)
        tool_with_ball_pcd = sampled_batched["tool_with_ball_pcd"].to(self.device)
        flow_pcd = sampled_batched["pcd_with_flow"].to(self.device)

        
        #binary 
        
        #label
        spillage = sampled_batched["binary_label"].to(self.device)

        level_1_reference = torch.tensor([1., 0.], device=self.device, dtype=torch.float32)
        level_1_indices = torch.where((spillage == level_1_reference).all(dim=1))[0]

        level_2_reference = torch.tensor([0., 1.], device=self.device, dtype=torch.float32)
        level_2_indices = torch.where((spillage == level_2_reference).all(dim=1))[0]
        collect_num = min(len(level_1_indices), len(level_2_indices))

        if collect_num == 0:
            print("wrong")
            if len(level_1_indices) == 0:
                level_2_indices = level_2_indices[torch.randperm(2)]
            else:
                level_1_indices = level_1_indices[torch.randperm(2)]
        else :
            level_1_indices = level_1_indices[torch.randperm(collect_num)]
            level_2_indices = level_2_indices[torch.randperm(collect_num)]
            
        # combined_indices = torch.cat((level_1_indices, level_2_indices, level_3_indices))
        combined_indices = torch.cat((level_1_indices, level_2_indices))
        '''

        
        # mul
        #label
        spillage = sampled_batched["spillage_type"].to(self.device)
        
        level_1_reference = torch.tensor([1., 0., 0.], device=self.device, dtype=torch.float32)
        level_1_indices = torch.where((spillage == level_1_reference).all(dim=1))[0]

        level_2_reference = torch.tensor([0., 1., 0.], device=self.device, dtype=torch.float32)
        level_2_indices = torch.where((spillage == level_2_reference).all(dim=1))[0]

        level_3_reference = torch.tensor([0., 0., 1.,], device=self.device, dtype=torch.float32)
        level_3_indices = torch.where((spillage == level_3_reference).all(dim=1))[0]

        # collect_num = int((len(level_2_indices) + len(level_3_indices))/2)
        collect_num = min(len(level_1_indices), len(level_2_indices), len(level_3_indices))

        # print(len(level_1_indices), len(level_2_indices), len(level_3_indices))

        level_1_indices = level_1_indices[torch.randperm(collect_num)]
        level_2_indices = level_2_indices[torch.randperm(collect_num)]
        level_3_indices = level_3_indices[torch.randperm(collect_num)]
        
        combined_indices = torch.cat((level_1_indices, level_2_indices, level_3_indices))
        '''



        eepose = eepose[combined_indices]
        spillage = spillage[combined_indices]
        tool_with_ball_pcd = tool_with_ball_pcd[combined_indices]
        ee_pcd = ee_pcd[combined_indices]
        flow_pcd = flow_pcd[combined_indices]
      
        
        

        pred_spillage = self.model(
                eepose, tool_with_ball_pcd, ee_pcd,flow_pcd,
            )

    
        loss = self.loss_function(pred_spillage.to(torch.float32), spillage.to(torch.float32))

        acc_num = 0
        
        for idx, pre_spillage in enumerate(pred_spillage):
  
            gd_class = torch.argmax(spillage[idx]).item()
            pre_class = torch.argmax(pre_spillage).item()
            self.gd_list[gd_class] += 1
            
            if gd_class == pre_class :            
                acc_num += 1
                self.acc_list[gd_class] += 1
        

        
        return (
            loss,
            acc_num / len(pred_spillage),
            pred_spillage,
            spillage,
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

        self.dataloaders["train"] = DataLoader(
            self.datasets["train"],
            batch_size=self.configs["batch_size"],
            num_workers=self.configs["num_workers"],
            pin_memory=True,
            drop_last=True,
            shuffle = True
        )

        self.dataloaders["val"] = DataLoader(
            self.datasets["val"],
            batch_size=self.configs["batch_size"],
            num_workers=self.configs["num_workers"],
            pin_memory=True,
            drop_last=True,
            shuffle = True
        )


        self.len_data = len(self.dataloaders["train"])
        self.val_len_data = len(self.dataloaders["val"])

        print("Finished setting up date")

    