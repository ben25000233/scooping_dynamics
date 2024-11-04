import h5py
import os
import numpy as np
from tqdm import tqdm
import torch
import open3d as o3d

# Path to the folder containing the .h5 files
# folder_path = "/media/hcis-s22/data/collected_data_with_pcd/dataset"
folder_path = "/home/hcis-s22/benyang/scoop-env/dynamics/new_dataset"

# Get a list of all .h5 files in the folder
h5_files = [f for f in os.listdir(f"{folder_path}/dataset") if f.endswith('.h5')]

tool_centroids_list = []
tool_distances_list = []
tool_with_ball_centroids_list = []
tool_with_ball_distances_list = []

def check_pcd_color(pcd):
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

def nor_pcd(pcd):
 
    nor_pcd = pcd[..., :3]
    centroids = np.mean(nor_pcd, axis=0, keepdims=True)
    

    # Normalize each point cloud by subtracting centroid and dividing by max distance
    nor_pcd -= centroids
    max_distances = np.max(np.sqrt(np.sum(nor_pcd**2, axis=1)), axis=0, keepdims=True)

    return centroids, max_distances


def align_point_cloud(points, target_points=3000):
    num_points = len(points)

    if num_points >= target_points:
        # Randomly downsample to target_points
        indices = np.random.choice(num_points, target_points, replace=False)
        indices = np.sort(indices)

    else:
        # Resample with replacement to reach target_points
        indices = np.random.choice(num_points, target_points, replace=True)
        indices = np.sort(indices)

    new_pcd = np.asarray(points)[indices]
    
    return new_pcd

# Iterate over each file and read it

for i, file_name in enumerate(tqdm(h5_files)):
    file_path = f"{folder_path}/dataset/{file_name}"
    with h5py.File(file_path, 'a') as dataset:  # 'a' mode opens for read/write without truncating
        info = f"{folder_path}/dataset_info/{file_name[:-3]}/spillage_type"
        with open(info, 'r') as file:
            content = file.read().replace('[', '').replace(']', '').replace(',', '')
    
        # Convert the cleaned content into a list of lists
        data_list = [list(map(int, line.split())) for line in content.strip().splitlines()]

        # Convert to NumPy array
        spillage_level = np.array(data_list)

        for j in range(10) :
            # Print the data and its shape
            print("Data:\n", spillage_level[0][10 * j: 10 * (j+1)])
            print(np.array(dataset["spillage_vol"][10 * j: 10 * (j+1)]))
            print(sum(dataset["spillage_vol"][10 * j: 10 * (j+1)]))
            print()
        exit()


        
        spillage_type = np.loadtxt(info, dtype=int)
        print(spillage_type.shape)
        # print(info)
        # print(file_name)
        exit()
        print()
        print(np.array(dataset["spillage_vol"][50:100]))
        print(np.array(dataset["spillage_type"][50:100]))
        exit()
        
        # front_pcd = dataset["top_pcd_point"]

        # Access the data inside the HDF5 file

        
        # # Extract hand_seg and top_seg once to avoid repeated access
        # hand_seg = dataset["hand_seg"]
        # top_seg = dataset["top_seg"]

        # # Precompute shapes
        # hand_shape = hand_seg.shape  # Assuming shape is (N, H, W)
        # top_shape = top_seg.shape  # Assuming shape is (N, H, W)

        # # Pre-allocate memory for correct_hand_seg and correct_front_seg
        # correct_hand_seg = np.empty((hand_shape[0], (hand_shape[1]-20) * hand_shape[2], 1))
        # correct_front_seg = np.empty((top_shape[0], top_shape[1] * (top_shape[2]-5), 1))

        # # Vectorized slicing, transposing, and reshaping
        # wrong_hand_seg = hand_seg[:, :-20, :]
        # correct_hand_seg[:] = wrong_hand_seg.transpose(0, 2, 1).reshape(hand_shape[0], -1, 1)

        # wrong_front_seg = top_seg[:, :, 5:]
        # correct_front_seg[:] = wrong_front_seg.transpose(0, 2, 1).reshape(top_shape[0], -1, 1)


        # spillage_type = np.array(dataset["spillage_type"])

        # # Initialize an empty array to hold the results
        # trans_spillage = np.zeros((len(spillage_type), 3))

        # # Apply conditions vectorized
        # trans_spillage[(spillage_type == 1) | (spillage_type == 2), 1] = 1
        # trans_spillage[(spillage_type == 3) | (spillage_type == 4), 2] = 1
        # trans_spillage[~((spillage_type == 1) | (spillage_type == 2) | (spillage_type == 3) | (spillage_type == 4)), 0] = 1

        # top_pcd = dataset["top_pcd_point"]
        # front_seg = dataset["front_seg"]

    
        # tool_pcd_list = []
        # tool_with_ball_pcd_list = []

        # for i in range(top_pcd.shape[0]):
            
        #     tool_indices = np.where(front_seg[i] == 1)[0]
        #     ball_indices = np.where(front_seg[i] == 2)[0]
        #     filter_index = np.concatenate((tool_indices, ball_indices), axis=0)

        #     tool_pcd = np.concatenate([top_pcd[i][tool_indices], front_seg[i][tool_indices]], axis = 1)
        #     tool_with_ball_pcd = np.concatenate([top_pcd[i][filter_index], front_seg[i][filter_index]], axis = 1)
          

        #     tool_pcd = align_point_cloud(tool_pcd)
        #     tool_with_ball_pcd = align_point_cloud(tool_with_ball_pcd)

        #     tool_centroids, tool_distances = nor_pcd(tool_pcd)
        #     tool_with_ball_centroids, tool_with_ball_distances = nor_pcd(tool_with_ball_pcd)
            
        #     tool_centroids_list.append(tool_centroids)
        #     tool_distances_list.append(tool_distances)
        #     tool_with_ball_centroids_list.append(tool_with_ball_centroids)
        #     tool_with_ball_distances_list.append(tool_with_ball_distances)

            # tool_pcd_list.append(tool_pcd)
            # tool_with_ball_pcd_list.append(tool_with_ball_pcd)

        # tool_pcd_list = np.array(tool_pcd_list)
        # tool_with_ball_pcd_list = np.array(tool_with_ball_pcd_list)

        # spillage_type = np.array(dataset["spillage_type"])
        # tensor_combined = np.stack((spillage_type[:, 0], spillage_type[:, 1] + spillage_type[:, 2]), axis=1)
 

        # Delete old datasets if they exist
        # if "tool_pcd" in dataset:
        #     del dataset["tool_pcd"]
        # if "tool_ball_pcd" in dataset:
        #     del dataset["tool_ball_pcd"]
        # del dataset["front_seg"]
        # del dataset["hand_depth"]
        # del dataset["hand_pcd_point"]
        # del dataset["hand_seg"]
        # del dataset["scoop_amount"]
        # del dataset["scoop_type"]
        # del dataset["scoop_vol"]
        # del dataset["top_depth"]


        # Create new datasets with corrected data
        # dataset.create_dataset("binary_label", data=np.array(tensor_combined))
        
        # dataset.create_dataset("tool_ball_pcd", data=np.array(tool_with_ball_pcd_list))
        

# mean_tool_centroids = sum(tool_centroids_list) / len(tool_centroids_list)
# mean_tool_distances = sum(tool_distances_list) / len(tool_distances_list)
# mean_tool_with_ball_centroids = sum(tool_with_ball_centroids_list) / len(tool_with_ball_centroids_list)
# mean_tool_with_ball_distances = sum(tool_with_ball_distances_list) / len(tool_with_ball_distances_list)

# pcd_nor_info = np.array([mean_tool_centroids[0], mean_tool_distances, mean_tool_with_ball_centroids[0], mean_tool_with_ball_distances])
# np.save("pcd_info", pcd_nor_info)



'''
original_file_path = "/media/hcis-s22/data/collected_data_with_pcd/dataset"
new_file_path = "/media/hcis-s22/data/new_data/dataset"

# Ensure the output directory exists
os.makedirs(new_file_path, exist_ok=True)

# Keys to keep in the new file
keys_to_keep = ["binary_label", "eepose", "spillage_amount", "spillage_type", "spillage_vol","tool_ball_pcd",  "tool_pcd","top_pcd_point" ]  # replace with actual keys you want to keep

# List all HDF5 files in the original directory
h5_files = [f for f in os.listdir(original_file_path) if f.endswith('.h5')]

for file_name in tqdm(h5_files):
    # Define paths for the original and new files
    file_path = os.path.join(original_file_path, file_name)
    new_file_name = os.path.join(new_file_path, file_name)  # Saving with the same name

    with h5py.File(file_path, "r") as original_file, h5py.File(new_file_name, "w") as new_file:
        # Iterate over keys to keep in the original file
        for key in keys_to_keep:
            if key in original_file:
                # Copy the desired datasets/groups to the new file
                original_file.copy(key, new_file)

print("Files created in the new directory with specified keys only.")

'''
