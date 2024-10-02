import h5py
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d


class MultimodalManipulationDataset(Dataset):
    """Multimodal Manipulation dataset with lazy loading."""

    def __init__(
        self,
        filename_list,
        data_length=50,
        training_type="selfsupervised",
        action_dim=7,
        single_env_steps=None, 
        type="train",
    ):
        """
        Args:
            filename_list (list): List of paths to HDF5 files.
            data_length (int): Length of the data sequence minus one.
            training_type (str): Type of training (e.g., selfsupervised).
            action_dim (int): Dimension of the action space.
            single_env_steps (int): Not used in this implementation.
            type (str): Type of dataset (e.g., 'train' or 'test').
        """
        self.single_env_steps = single_env_steps
        self.dataset_path = filename_list
        self.data_length_in_eachfile = data_length - 1
        self.training_type = training_type
        self.action_dim = action_dim
        self.type = type


        # We no longer load all the data at once
        self.file_handles = [h5py.File(file, 'r') for file in filename_list]
    

    def __len__(self):
        return len(self.dataset_path) * self.data_length_in_eachfile

    def __getitem__(self, idx):
        # Determine which file and which data entry within that file to load
        file_idx = idx // self.data_length_in_eachfile
        data_index = idx % self.data_length_in_eachfile

        # Open the corresponding file and load the specific data entry
        dataset = self.file_handles[file_idx]
        data = self._read_data_from_file(dataset, data_index)

        return data

    def _read_data_from_file(self, dataset, idx):
      
        # Read data from a single file for the given index

        # single_num : 8
        single_num = len(dataset["hand_seg"]) / (self.data_length_in_eachfile + 1)

        current_index = int(idx * single_num) 

        # total predict prame : look back add current frame
        look_back_frame = 2
        # train with history(need to modify)
     
        
        if current_index == 0:

            hand_seg = dataset["hand_seg"][0]
            front_seg = dataset["front_seg"][0]
          
          
            hand_depth = np.tile(dataset["hand_depth"][0], (look_back_frame+1, 1, 1)).astype(np.float32)
            front_depth = np.tile(dataset["top_depth"][0], (look_back_frame+1, 1, 1)).astype(np.float32)
            hand_pcd = np.tile(dataset["hand_pcd_point"][0], (look_back_frame+1, 1, 1)).astype(np.float32)
            front_pcd = np.tile(dataset["top_pcd_point"][0], (look_back_frame+1, 1, 1)).astype(np.float32)
            hand_seg = np.tile(hand_seg, (look_back_frame+1, 1, 1)).astype(np.float32)
            front_seg = np.tile(front_seg, (look_back_frame+1, 1, 1)).astype(np.float32)
        
            
        else : 
            begin_idx = current_index - look_back_frame 
            end_idx = current_index + 1

            hand_depth = dataset["hand_depth"][begin_idx:end_idx].astype(np.float32)
            front_depth = dataset["top_depth"][begin_idx:end_idx].astype(np.float32)
     
            

            hand_pcd = dataset["hand_pcd_point"][begin_idx:end_idx].astype(np.float32)
            front_pcd = dataset["top_pcd_point"][begin_idx:end_idx].astype(np.float32)
      

            hand_seg = dataset["hand_seg"][begin_idx:end_idx].astype(np.float32)
            front_seg = dataset["front_seg"][begin_idx:end_idx].astype(np.float32)


 
        hand_pcd = np.concatenate((hand_pcd, hand_seg), axis=2)
        front_pcd = np.concatenate((front_pcd, front_seg), axis=2)

        #front_pcd

        filter_front_pcd = []

        for i in range(front_pcd.shape[0]):
            
            tool_indices = np.where(front_seg[i] == 1)[0]
            ball_indices = np.where(front_seg[i] == 2)[0]
            filter_index = np.concatenate((tool_indices, ball_indices), axis=0)
            unallign_pcd = front_pcd[i][filter_index]
            seg_pcd = self.align_point_cloud(unallign_pcd)

   
            filter_front_pcd.append(seg_pcd)
        filter_front_pcd = np.array(filter_front_pcd)

        # future ee_pose
        future_eepose_num = 7 # in data collectoin, 8 eepose(current and future 7 step) related to a spillage => future eepose <= 7
        current_ee_index = int((idx + 1) * single_num)+1
        target_eepose = current_ee_index + future_eepose_num
        eepose = dataset["eepose"][current_ee_index: target_eepose]

 
        #future ee_pcd

        front_pcd = dataset["top_pcd_point"][current_ee_index : target_eepose].astype(np.float32)
        front_seg = dataset["front_seg"][current_ee_index : target_eepose].astype(int)
        
  
        ee_pcd = []
        for i in range(future_eepose_num):
        
            filter_index = np.where(front_seg[i] == 1)[0]
            pcd = np.concatenate((front_pcd[i], front_seg[i]), axis=1)
            unallign_pcd = pcd[filter_index]
            seg_pcd = self.align_point_cloud(unallign_pcd)
            # self.check_pcd_color(seg_pcd)
            
            ee_pcd.append(seg_pcd)

            
        ee_pcd = np.array(ee_pcd)
 
        spillage_index = dataset["spillage_type"][idx]
        scoop_index = dataset["scoop_type"][idx]

        # binary label
        # if spillage_index == 0 :
        #     spillage_amount = 0
        # else :
        #     spillage_amount = 1

        # if scoop_index == 0 :
        #     scoop_amount = 0
        # else :
        #     scoop_amount = 1

        # multi-label
        # Create a one-hot encoded vector
        # spillage_amount = np.zeros(3)
        # if spillage_index == 1 or spillage_index == 2 :
        #     spillage_amount[1] = 1
        # elif spillage_index == 3 or spillage_index == 4 :
        #     spillage_amount[2] = 1
        # else :
        #     spillage_amount[0] = 1

        # scoop_amount = np.zeros(6)
        # scoop_amount[scoop_index] = 1

      

        single_data = {
            "hand_depth": hand_depth,
            "hand_seg": hand_seg,
            "eepose": eepose,
            "ee_pcd" : ee_pcd,
            "spillage_vol": spillage_index,
            # "scoop_vol": scoop_amount,
            "front_depth" : front_depth,
            "front_seg" : front_seg, 
            "hand_pcd" : hand_pcd,
            "front_pcd" : filter_front_pcd,
        }
    

        return single_data

    def check_pcd_color(self, pcd):
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
            points.append(pcd[i][:3])
            colors.append(color_map[pcd[i][3]])

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([point_cloud])
        

    def __del__(self):
        # Close all file handles when the dataset object is deleted
        for file_handle in self.file_handles:
            file_handle.close()

    def align_point_cloud(self, points, target_points=3000):
        num_points = len(points)
        if num_points >= target_points:
            # Randomly downsample to target_points
            indices = np.random.choice(num_points, target_points, replace=False)
        else:
            # Resample with replacement to reach target_points
            indices = np.random.choice(num_points, target_points, replace=True)

        new_points = np.asarray(points)[indices]
        return new_points
