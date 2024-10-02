import h5py
import os
import numpy as np
from tqdm import tqdm

# Path to the folder containing the .h5 files
folder_path = './../../collected_data_with_pcd/dataset'

# Get a list of all .h5 files in the folder
h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]

# Iterate over each file and read it
for i, file_name in enumerate(tqdm(h5_files)):
    file_path = os.path.join(folder_path, file_name)
    with h5py.File(file_path, 'a') as dataset:  # 'a' mode opens for read/write without truncating
        # Access the data inside the HDF5 file

        
        # Extract hand_seg and top_seg once to avoid repeated access
        hand_seg = dataset["hand_seg"]
        top_seg = dataset["top_seg"]

        # Precompute shapes
        hand_shape = hand_seg.shape  # Assuming shape is (N, H, W)
        top_shape = top_seg.shape  # Assuming shape is (N, H, W)

        # Pre-allocate memory for correct_hand_seg and correct_front_seg
        correct_hand_seg = np.empty((hand_shape[0], (hand_shape[1]-20) * hand_shape[2], 1))
        correct_front_seg = np.empty((top_shape[0], top_shape[1] * (top_shape[2]-5), 1))

        # Vectorized slicing, transposing, and reshaping
        wrong_hand_seg = hand_seg[:, :-20, :]
        correct_hand_seg[:] = wrong_hand_seg.transpose(0, 2, 1).reshape(hand_shape[0], -1, 1)

        wrong_front_seg = top_seg[:, :, 5:]
        correct_front_seg[:] = wrong_front_seg.transpose(0, 2, 1).reshape(top_shape[0], -1, 1)


        spillage_type = np.array(dataset["spillage_type"])

        # Initialize an empty array to hold the results
        trans_spillage = np.zeros((len(spillage_type), 3))

        # Apply conditions vectorized
        trans_spillage[(spillage_type == 1) | (spillage_type == 2), 1] = 1
        trans_spillage[(spillage_type == 3) | (spillage_type == 4), 2] = 1
        trans_spillage[~((spillage_type == 1) | (spillage_type == 2) | (spillage_type == 3) | (spillage_type == 4)), 0] = 1


        
        # Delete old datasets if they exist
        if "hand_seg" in dataset:
            del dataset["hand_seg"]
        if "top_seg" in dataset:
            del dataset["top_seg"]
        # Create new datasets with corrected data
        dataset.create_dataset("hand_seg", data=np.array(correct_hand_seg))
        dataset.create_dataset("front_seg", data=np.array(correct_front_seg))
        

        if "spillage_type" in dataset:
            del dataset["spillage_type"]
        dataset.create_dataset("spillage_type", data=np.array(trans_spillage))



        
