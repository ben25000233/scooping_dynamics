import h5py
import numpy as np
from tqdm import tqdm
import open3d as o3d

from torch.utils.data import Dataset


class MultimodalManipulationDataset(Dataset):
    """Multimodal Manipulation dataset."""

    def __init__(
        self,
        filename_list,
        data_length=50,
        training_type="selfsupervised",
        action_dim=7,
        single_env_steps = None, 
        type = "train",
    ):
        """
        Args:
            hdf5_file (handle): h5py handle of the hdf5 file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.single_env_steps = single_env_steps
        self.dataset_path = filename_list
        self.data_length_in_eachfile = data_length
        self.training_type = training_type
        self.datasets = {}
        self.action_dim = action_dim
        self.type = type
        self.dataset = []

        self._load_data()


    def __len__(self):
        length = len(self.dataset_path) * self.data_length_in_eachfile 

        return length
    
    def __getitem__(self, idx):

        file_idx = idx // self.data_length_in_eachfile
        data_index = idx % self.data_length_in_eachfile
        
        return self.dataset[file_idx][data_index]

        
    def _load_data(self):
        # Load data from all files
        print(f"Load {self.type} data...")
        for idx in tqdm(range(len(self.dataset_path))):
            with h5py.File(self.dataset_path[idx], "r", swmr=True, libver="latest") as dataset:
                data = self._read_data_from_file(dataset, self.dataset_path[idx])
                self.dataset.append(data)

    def _read_data_from_file(self, dataset, file_name):
        # Read data from a single file and return it as a list
        data = []
        file_num = int(file_name.split('_')[-1][:-3])

        for idx in range(self.data_length_in_eachfile):

            pcd_env_num = idx // self.single_env_steps
            pcd_index = idx % self.single_env_steps

            property = dataset["property"][:]
            eepose = dataset["eepose"][idx]
            next_eepose = dataset["next_eepose"][idx]
            spillage = dataset["spillage_amount"][idx]
            
            single_data = {
                "pcd_info": (file_num, pcd_env_num, pcd_index),
                "property": property,
                "eepose": eepose,
                "next_eepose": next_eepose,
                "spillage_amount": spillage,
            }

            data.append(single_data)

        return data
