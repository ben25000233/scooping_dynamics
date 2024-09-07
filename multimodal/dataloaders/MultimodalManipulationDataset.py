import h5py
import numpy as np
from torch.utils.data import Dataset


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

        all_eepose = dataset["eepose"]
        current_index = int(idx * single_num) 

        # total predict prame : look back add current frame
        look_back_frame = 5
        # train with history(need to modify)
        
        if current_index == 0:
            hand_depth = np.tile(dataset["hand_depth"][0], (look_back_frame+1, 1, 1)).astype(np.float32)
            hand_seg = np.tile(dataset["hand_seg"][0], (look_back_frame+1, 1, 1)).astype(np.float32)
            front_depth = np.tile(dataset["top_depth"][0], (look_back_frame+1, 1, 1)).astype(np.float32)
            # hand_pcd = np.tile(dataset["hand_pcd"][0], (look_back_frame+1, 1, 1)).astype(np.float32)
            # top_pcd = np.tile(dataset["top_pcd"][0], (look_back_frame+1, 1, 1)).astype(np.float32)
            
        else : 
            begin_idx = current_index - look_back_frame 
            end_idx = current_index + 1
            hand_depth = dataset["hand_depth"][begin_idx:end_idx].astype(np.float32)
            hand_seg = dataset["hand_seg"][begin_idx:end_idx].astype(np.float32)
            front_depth = dataset["top_depth"][begin_idx:end_idx].astype(np.float32)

            # hand_pcd = dataset["hand_pcd"][begin_idx:end_idx].astype(np.float32)
            # top_pcd = dataset["top_pcd"][begin_idx:end_idx].astype(np.float32)
       
        eepose = all_eepose[int((idx + 1) * single_num):int((idx + 1) * single_num) + 8]

        spillage_index = dataset["spillage_type"][idx]
        scoop_index = dataset["scoop_type"][idx]


  
        # Create a one-hot encoded vector
        spillage_amount = np.zeros(5)
        spillage_amount[spillage_index] = 1

        scoop_amount = np.zeros(6)
        scoop_amount[scoop_index] = 1

        # normalized_depth = []
        # normalized_seg = []

        # for img in hand_depth:
        #     min_val = np.min(img)
        #     max_val = np.max(img)
        #     # Normalize the image to 0-1 range
        #     normalized_img = (img - min_val) / (max_val - min_val)
        #     # Append the normalized image to the list
        #     normalized_depth.append(normalized_img)
        # depth_arr = np.array(normalized_depth)

        # for img in hand_seg:
        #     min_val = np.min(img)
        #     max_val = np.max(img)
        #     # Normalize the image to 0-1 range
        #     normalized_img = (img - min_val) / (max_val - min_val)
        #     # Append the normalized image to the list
        #     normalized_seg.append(normalized_img)
        # seg_arr = np.array(normalized_seg)
            

        single_data = {
            "hand_depth": hand_depth,
            "hand_seg": hand_seg,
            "eepose": eepose,
            "spillage_vol": spillage_amount,
            "scoop_vol": scoop_amount,
            "front_depth" : front_depth,
            # "hand_pcd" : hand_pcd,
            # "front_pcd" : top_pcd,
        }

        return single_data

    def __del__(self):
        # Close all file handles when the dataset object is deleted
        for file_handle in self.file_handles:
            file_handle.close()
