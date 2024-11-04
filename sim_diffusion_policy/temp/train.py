from pyconfigparser import configparser
from torch.utils.data.dataset import Dataset
import glob
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from tqdm import tqdm
import sys
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import math
import os
import numpy as np
from model import DiffusionPolicy, EMAModel, SingleObEncoder
from argparse import ArgumentParser
import copy
from torch.utils.tensorboard import SummaryWriter
import torchinfo


class FoodDataset(Dataset):
    def __init__(self, cfg, stage):
        self.rootPath = cfg.data.rootPath
        if stage=='train':
            self.obs = sorted(glob.glob(self.rootPath + '/ob_front*.pt'))
            self.traj = sorted(glob.glob(self.rootPath + '/traj*.pt'))
        else:
            temp_obs = sorted(glob.glob(self.rootPath + '/ob_front*.pt'))
            temp_traj = sorted(glob.glob(self.rootPath + '/traj*.pt'))
            self.obs = temp_obs[30:33]
            self.traj = temp_traj[30:33]    

    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, index):
        obs = torch.load(self.obs[index])
        traj = torch.load(self.traj[index])
        return obs, traj

def train(cfg, 
          train_dataloader, val_dataloader,
          diffusion_model, ema, ema_model, 
          optimizer, scheduler,
          output_dir):
    
    since = time.time()
    save_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
   
    writer = SummaryWriter('./runs')    
    
    for epoch in range(cfg.training.num_epochs):
        print('Epoch {}/{}'.format(epoch+1, cfg.training.num_epochs))
        print('-' * 10)

        # ---------------------------- Training -------------------------------
        diffusion_model.train()
        train_losses = list()

        for obs, traj in tqdm(train_dataloader):
            # to device
            obs = obs.to(cfg.training.device, dtype=torch.float32)
            traj = traj.to(cfg.training.device, dtype=torch.float32)

            optimizer.zero_grad()
            loss = diffusion_model.compute_loss((obs, traj))

            loss.backward()
            optimizer.step()
            scheduler.step()
            ema_model.step(diffusion_model)

            # statistics
            train_losses.append(loss.item())    
        
        epoch_loss = np.mean(train_losses)
        print("lr: {}".format(scheduler.optimizer.param_groups[0]['lr']))
        print("train Loss: {:.9f}".format(epoch_loss))
        # print("")

        # add to tensorboard
        writer.add_scalar('training loss', epoch_loss, epoch+1)

        if (epoch+1)%cfg.training.save_epoch==0:
            torch.save({
                'epoch': epoch,
                'dp_state_dict': diffusion_model.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                }, os.path.join(save_dir, 'epoch={}.pth'.format(epoch+1)))
            
        # ---------------------------- Inference -------------------------------
        policy = ema
        policy.eval()
        val_losses = list()

        with torch.no_grad():
            for obs, traj in tqdm(val_dataloader):
                # to device
                obs = obs.to(cfg.training.device, dtype=torch.float32)
                traj = traj.to(cfg.training.device, dtype=torch.float32)
                _, pred_action = policy.predict_action((obs, traj))
                # statistics
                mse = torch.nn.functional.mse_loss(pred_action, traj)
                # val_losses += mse.item()
                val_losses.append(mse.item())

        val_epoch_loss = np.mean(val_losses)
        print("val MSE: {:.9f}".format(val_epoch_loss))
        print("")

        # add to tensorboard
        writer.add_scalar('val MSE', val_epoch_loss, epoch+1)
            
        # uodate
        scheduler.step(val_epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    torch.save({
                'epoch': epoch,
                'dp_state_dict': diffusion_model.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                }, os.path.join(save_dir, 'epoch={}.pth'.format(epoch+1)))


class Print_Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def main():

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", default="grits.yaml")
    args = parser.parse_args()
    cfg = configparser.get_config(file_name=args.config)

    #--------------------#
    # create save folder #
    #--------------------#
    output_dir = "experiments/{}".format(cfg.trial_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    sys.stdout = Print_Logger(os.path.join(output_dir, "log.txt"))

    #-----------------#
    # prepare dataset #
    #-----------------#
    train_set = FoodDataset(cfg, stage='train')
    train_dataloader = DataLoader(
        dataset=train_set, 
        batch_size=cfg.train_dataloader.batch_size, 
        num_workers=cfg.train_dataloader.num_workers, 
        shuffle=True,
        pin_memory=False,
        persistent_workers=False)
    val_set = FoodDataset(cfg, stage='val')
    val_dataloader = DataLoader(
        dataset=val_set, 
        batch_size=cfg.val_dataloader.batch_size, 
        num_workers=cfg.val_dataloader.num_workers, 
        shuffle=True,
        pin_memory=False,
        persistent_workers=True)

    #---------#
    #  model  #
    #---------#
    obs_encoder = SingleObEncoder(cfg)
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type='epsilon'
    )
    diffusion_model = DiffusionPolicy(
        cfg,
        obs_encoder,
        noise_scheduler
    )
    diffusion_model.to(cfg.training.device) 
    ema = copy.deepcopy(diffusion_model)
    ema.to(cfg.training.device)
    ema_model = EMAModel(ema)

    #---------#
    # setting #
    #---------#    
    # def lr_lambda(current_step):
    #     # linear
    #     num_training_steps=len(train_dataloader) * cfg.training.num_epochs 
    #     if current_step < cfg.training.lr_warmup_steps:
    #         return float(current_step) / float(max(1, cfg.training.lr_warmup_steps))
    #     return max(
    #         0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - cfg.training.lr_warmup_steps))
    #     )  
    def lr_lambda(current_step):
        num_training_steps=len(train_dataloader) * cfg.training.num_epochs # 293*600        
        if current_step < cfg.training.lr_warmup_steps:
            return float(current_step) / float(max(1, cfg.training.lr_warmup_steps))
        progress = float(current_step - cfg.training.lr_warmup_steps) / float(max(1, num_training_steps - cfg.training.lr_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(cfg.training.num_cycles) * 2.0 * progress)))
    optimizer = optim.AdamW(
        diffusion_model.parameters(), 
        lr=cfg.training.lr, betas=(0.95, 0.999), 
        eps=1.0e-8, 
        weight_decay=1.0e-6
    )
    scheduler = lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda, 
    )
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, min_lr=0.000001)

    # load pretrained
    # checkpoint = torch.load('experiments/model_ee_linear/ckpt/epoch=600.pth')
    # diffusion_model.load_state_dict(checkpoint['dp_state_dict'])
    # ema.load_state_dict(checkpoint['ema_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # torch.cuda.empty_cache()

    #----------#
    # training #
    #----------# 
    train(cfg, 
          train_dataloader, val_dataloader,
          diffusion_model, ema, ema_model,
          optimizer, scheduler,
          output_dir)
    
    
    # print(len(train_dataloader))
    # num_steps = len(train_dataloader) * cfg.training.num_epochs
    # learning_rates = [cfg.training.lr * lr_lambda(step) for step in range(num_steps)]
    # import matplotlib.pyplot as plt
    # # Plot the learning rate schedule
    # plt.figure(figsize=(10, 6))
    # plt.plot(learning_rates)
    # plt.title('Learning Rate Schedule Over Training Steps')
    # plt.xlabel('Training Step')
    # plt.ylabel('Learning Rate')
    # plt.grid(True)
    # plt.show()


# tensorboard --logdir /home/yling/Desktop/GRITS/runs --host=127.0.0.1 --port=6007
if __name__ == '__main__':
    main()