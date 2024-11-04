

import numpy as np
from pynput.keyboard import Listener, KeyCode
import math
import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# from tf.transformations import euler_from_quaternion, quaternion_from_euler
import time
from rgbd_model import SingleObEncoder, DiffusionPolicy, EMAModel, RotationTransformer
# from torch_ema import ExponentialMovingAverage
import copy

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

import open3d as o3d
# from dt_apriltags import Detector
import quaternion
from pytorch3d.transforms import quaternion_to_matrix
import transforms3d
import matplotlib.pyplot as plt




class LfD():
    def __init__(self):

        # normalize
        input_range = torch.load('../dataset/input_range.pt')
        self.input_max = input_range[0,:]
        self.input_min = input_range[1,:]
        self.input_mean = input_range[2,:]

        self.config_file = "./config/grits.yaml"
        import yaml
        with open(self.config_file, 'r') as file:
            cfg = yaml.safe_load(file)


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
        self.diffusion_model = DiffusionPolicy(
            cfg,
            obs_encoder,
            noise_scheduler
        )

        self.diffusion_model.to(cfg["training"]["device"])
   
        checkpoint = torch.load('rgbd_500.pth', map_location="cuda:0")
  
        self.diffusion_model.load_state_dict(checkpoint['dp_state_dict'])
        
    
     
    def set_stiffness(self, k_t1, k_t2, k_t3,k_r1,k_r2,k_r3, k_ns):
        set_K = dynamic_reconfigure.client.Client('/dynamic_reconfigure_compliance_param_node', config_callback=None)
        set_K.update_configuration({"translational_stiffness_X": k_t1})
        set_K.update_configuration({"translational_stiffness_Y": k_t2})
        set_K.update_configuration({"translational_stiffness_Z": k_t3})        
        set_K.update_configuration({"rotational_stiffness_X": k_r1}) 
        set_K.update_configuration({"rotational_stiffness_Y": k_r2}) 
        set_K.update_configuration({"rotational_stiffness_Z": k_r3})
        set_K.update_configuration({"nullspace_stiffness": k_ns})   
    
    def ee_pos_callback(self, data):
        self.curr_pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.curr_ori = np.array([data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z])

    def ft_callback(self, data):
        self.ft_f = np.array([data.wrench.force.x, data.wrench.force.y, data.wrench.force.z])
        self.ft_t = np.array([data.wrench.torque.x, data.wrench.torque.y, data.wrench.torque.z])
    
    def rgb_front_callback(self, data):
        bridge = CvBridge()
        self.rgb_front_image = cv2.flip(cv2.resize(np.array(bridge.imgmsg_to_cv2(data, data.encoding)), (640, 480)), -1)

    def depth_front_callback(self, data):
        bridge = CvBridge()
        self.depth_front_image = cv2.flip(cv2.resize(np.array(bridge.imgmsg_to_cv2(data, data.encoding)), (640, 480)), -1)

    def rgb_ee_callback(self, data):
        bridge = CvBridge()
        self.rgb_ee_image = cv2.flip(cv2.resize(np.array(bridge.imgmsg_to_cv2(data, data.encoding)), (640, 480)), -1)

    def depth_ee_callback(self, data):
        bridge = CvBridge()
        self.depth_ee_image = cv2.flip(cv2.resize(np.array(bridge.imgmsg_to_cv2(data, data.encoding)), (640, 480)), -1)
    
    def rgb_demo_callback(self, data):
        bridge = CvBridge()
        self.rgb_demo_image = cv2.flip(cv2.resize(np.array(bridge.imgmsg_to_cv2(data, data.encoding)), (640, 480)), -1)

    def depth_demo_callback(self, data):
        bridge = CvBridge()
        self.depth_demo_image = cv2.flip(cv2.resize(np.array(bridge.imgmsg_to_cv2(data, data.encoding)), (640, 480)), -1)
    
    def base_callback(self, data):
        self.init_force_mat = np.array(data.data).reshape((16, 3))
    
    def _on_press(self, key):
        if key == KeyCode.from_char('e'):
            self.end = True   
        elif key == KeyCode.from_char('s'):
            self.tmp_stop = True
    
    def to_eular(self, pose):
        # ori [w,x,y,z]
        # orientation_list = [pose[4], pose[5], pose[6], pose[3]]
        (roll, pitch, yaw) = transforms3d.euler.quat2euler(pose[3:])
        return np.array([pose[0], pose[1], pose[2], roll, pitch, yaw])
    
    def to_qua(self, pose):
        # ori [roll,pitch,yaw]
        orientation_list = [pose[3], pose[4], pose[5]]
        q = quaternion_from_euler(orientation_list)
        return np.array([pose[0], pose[1], pose[2], q[0], q[1], q[2], q[3]])
    
    # visualize pcd
    def depth_image_to_point_cloud(
        self, rgb, depth, intrinsic_matrix, depth_scale=1, remove_outliers=True, z_threshold=None, mask=None, device="cuda:0"):
        # process input
        rgb = torch.from_numpy(np.array(rgb).astype(np.float32) / 255).to(device)
        depth = torch.from_numpy(depth.astype(np.float32)).to(device)
        intrinsic_matrix = torch.from_numpy(intrinsic_matrix.astype(np.float32)).to(device)
        
        # depth image to point cloud
        h, w = depth.shape
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        x = x.float()
        y = y.float()

        ones = torch.ones((h, w), dtype=torch.float32)
        xy1s = torch.stack((x, y, ones), dim=2).view(w * h, 3).t()
        xy1s = xy1s.to(device)

        depth /= depth_scale
        points = torch.linalg.inv(intrinsic_matrix) @ xy1s
        points = torch.mul(depth.view(1, -1, w * h).expand(3, -1, -1), points.unsqueeze(1))
        points = points.squeeze().T

        colors = rgb.reshape(w * h, -1)
        
        # masks
        if mask is not None:
            mask = torch.from_numpy(mask).to(device)
            points = points[mask.reshape(-1), :]
            colors = colors[mask.reshape(-1), :]
        
        # remove far points
        if z_threshold is not None:
            valid = (points[:, 2] < z_threshold)
            points = points[valid]
            colors = colors[valid]

        # create o3d point cloud
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
        scene_pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())

        return np.asarray(scene_pcd.points), np.asarray(scene_pcd.colors), scene_pcd

    # capture apriltag and calculate goal pose for transfering and pouring
    def estimate_target_pose(self, rgb, depth, method='tag', vis=False):

        intrinsic = np.array([[606.2581787109375, 0.0, 322.64874267578125], [0.0, 606.0323486328125, 235.183349609375], [0.0, 0.0, 1.0]])
        extrinsic = np.load('dataset/cam2base.npy')

        if method=='tag':
            detector = Detector(
                families='tag36h11',
                nthreads=6,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0
            )
            rgb = rgb.astype(np.uint8, 'C')
            grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            camera_params = (intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2])
            tags = detector.detect(grey, estimate_tag_pose=True, camera_params=camera_params, tag_size=0.02)
            assert len(tags) > 0, "No tag detected"

            tag65_pose = np.eye(4)
            tag85_pose = np.eye(4)
            for tag in tags:
                if tag.tag_id==65:
                    tag65_pose[0:3, 0:3] = tag.pose_R
                    tag65_pose[0:3, 3] = tag.pose_t.reshape(-1)
                elif tag.tag_id==85:
                    tag85_pose[0:3, 0:3] = tag.pose_R
                    tag85_pose[0:3, 3] = tag.pose_t.reshape(-1)

            # trasnfer tag pose to world coordinate
            tag65_pose = np.dot(extrinsic, tag65_pose)
            tag85_pose = np.dot(extrinsic, tag85_pose)

            # obtain food container's center
            food_center_pose = tag85_pose
            food_center_pose[0, 3] -= 0.12
            food_center_pose[1, 3] -= 0.02
            food_center_pose[2, 3] += 0.2

            # trasnfer to goal pose
            goal_pose = tag65_pose
            goal_pose[0, 3] -= 0.12  # 0.12
            goal_pose[1, 3] -= 0.07 # -0.02
            goal_pose[2, 3] += 0.2

        if vis:
            # visualize
            _, _, pcd = self.depth_image_to_point_cloud(rgb, depth, intrinsic, depth_scale=1000)
            pcd.transform(extrinsic)
            bowl = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            bowl.transform(food_center_pose)
            goal = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            goal.transform(goal_pose)
            o3d.visualization.draw_geometries([pcd, goal])

        return food_center_pose[0:3, 3], goal_pose[0:3, 3]
    
    def _normalize(self, data, input_max, input_min, input_mean):
        ranges = input_max - input_min
        data_normalize = torch.zeros_like(data)
        for i in range(3):
            if ranges[i] < 1e-4:
                data_normalize[i] = data[i] - input_mean[i]
            else:
                data_normalize[i] = -1 + 2 * (data[i] - input_min[i]) / ranges[i]
        data_normalize[3:] = data[3:]
        return data_normalize
    
    # print 
    def print_ee_euler(self):
        while not rospy.is_shutdown():
            ee_show = self.to_eular(np.concatenate((self.curr_pos, self.curr_ori), axis = 0))
            print("ee_show= ", ee_show)
            self.control_rate.sleep()
    
    # ------ #
    # Homing #
    #--------#

    def go_to_start_pose(self, goal_pose, stage):
        rospy.wait_for_message("/cartesian_pose", PoseStamped)
        print("Go to start pose.")

        start = self.curr_pos
        start_ori = self.curr_ori

        goal_ = goal_pose[0:3]

        squared_dist = np.sum(np.subtract(start, goal_)**2, axis=0)
        dist = np.sqrt(squared_dist)
        print("dist", dist)
        if stage=='homing':
            interp_dist = 0.001  # 0.001 [m]
        else:
            interp_dist = 0.005
        step_num_lin = math.floor(dist / interp_dist)
        
        print("num of steps linear", step_num_lin)
        
        q_start = quaternion.quaternion(start_ori[0], start_ori[1], start_ori[2], start_ori[3])
        print("q_start", q_start)
        q_goal = quaternion.quaternion(goal_pose[3], goal_pose[4], goal_pose[5], goal_pose[6])

        inner_prod = q_start.x*q_goal.x + q_start.y*q_goal.y + q_start.z*q_goal.z + q_start.w*q_goal.w
        
        if inner_prod < 0:
            q_start.x=-q_start.x
            q_start.y=-q_start.y
            q_start.z=-q_start.z
            q_start.w=-q_start.w

        inner_prod = q_start.x*q_goal.x+q_start.y*q_goal.y+q_start.z*q_goal.z+q_start.w*q_goal.w
        theta = np.arccos(np.abs(inner_prod))
        interp_dist_polar = 0.001
        step_num_polar = math.floor(theta / interp_dist_polar)
        
        print("num of steps polar", step_num_polar)
        
        step_num = np.max([step_num_polar,step_num_lin])
        
        print("num of steps", step_num)
        x = np.linspace(start[0], goal_pose[0], step_num)
        y = np.linspace(start[1], goal_pose[1], step_num)
        z = np.linspace(start[2], goal_pose[2], step_num)
        
        goal = PoseStamped()
        
        goal.pose.position.x = x[0]
        goal.pose.position.y = y[0]
        goal.pose.position.z = z[0]
        
        quat=np.slerp_vectorized(q_start, q_goal, 0.0)
        goal.pose.orientation.x = quat.x
        goal.pose.orientation.y = quat.y
        goal.pose.orientation.z = quat.z
        goal.pose.orientation.w = quat.w

        self.ee_pub.publish(goal)
        self.set_stiffness(3500, 3500, 3500, 50, 50, 50, 0.0)

        self.control_rate.sleep()

        goal = PoseStamped()
        for i in range(step_num):
            print("i= ", i)
            now = time.time()         
            goal.header.seq = 1
            goal.header.stamp = rospy.Time.now()
            goal.header.frame_id = "map"

            goal.pose.position.x = x[i]
            goal.pose.position.y = y[i]
            goal.pose.position.z = z[i]
            quat = np.slerp_vectorized(q_start, q_goal, i/step_num)
            goal.pose.orientation.x = quat.x
            goal.pose.orientation.y = quat.y
            goal.pose.orientation.z = quat.z
            goal.pose.orientation.w = quat.w
            self.ee_pub.publish(goal)
            self.control_rate.sleep()
        
        print("Now pos: {}".format(self.curr_pos))
        ee_show = self.to_eular(np.concatenate((self.curr_pos, self.curr_ori), axis = 0))
        print("Now ori: {}".format(np.array([ee_show[3], ee_show[4], ee_show[5]])))
        print("")

        rgb = Image.fromarray(np.uint8(self.rgb_front_image))
        rgb.save('now.png')
    
    # ------- #
    # predict #
    #-------- #

    def rgb_transform(self, img):
        return transforms.Compose([
            transforms.Resize([240, 320]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])(img)

    def depth_transform(self, img):
        return transforms.Compose([
            transforms.Resize([240, 320]),
            transforms.ToTensor(),
        ])(img)
    
    def run_model(self, image_list, depth_list, eepose_list):
   
        rotation_transformer_forward = RotationTransformer('quaternion', 'rotation_6d')
        rotation_transformer_backward = RotationTransformer('rotation_6d', 'quaternion')

        policy = self.diffusion_model
        policy.eval()

        rgb_input = []
        depth_input = []
        ee_input = []

        start_guidance = False
        with torch.no_grad():
            step = 0
            for i in range(image_list.shape[0]):
               
                rgb_input.append(self.rgb_transform(Image.fromarray(image_list[i].astype('uint8'), 'RGB')))
                depth_PIL = depth_list[i].astype(np.float32)
                depth_input.append(self.depth_transform(Image.fromarray(depth_PIL / np.max(depth_PIL))))
                

            rgb_in = torch.stack(rgb_input, dim=0) # [5, 3, 240, 320]
            depth_in = torch.stack(depth_input, dim=0)  # [5, 1, 240, 320]
            
            obs_in = torch.unsqueeze(torch.cat((rgb_in, depth_in), 1), 0).to('cuda:0', dtype=torch.float32) # [1, 5, 4, 240, 320]
            
        
            # for i in range(len(depth_in)):
            #     print
            #     plt.imshow(depth_in[i].reshape(240, 320, 1))
            #     plt.show()
            
    
            action, _ = policy.predict_action((obs_in, eepose_list))  
 
            # transform rotate
            action_publish = action.cpu().detach().numpy().squeeze(0) # [8, 9]
            action_position = action_publish[:, 0:3]
            action_rotation = rotation_transformer_backward.forward(action_publish[:, 3:])
            action_publish = np.concatenate((action_position, action_rotation), -1) # [8, 7]
            

            euler_action = []
            for action in action_publish:
                
                euler = self.to_eular(action)
                euler_action.append(euler)
                # print(action, euler)
               
            euler_action = np.array(euler_action)

        return euler_action
              
        
# roslaunch realsense2_camera rs_camera.launch camera:=cam_front serial_no:=146322072196 align_depth:=True intial_reset:=True
if __name__ == '__main__':

    rospy.init_node('run_model', anonymous=True) 
    rospy.loginfo('started')
    rospy.sleep(1)
    input("press enter to start.")

    lfd = LfD()
    start_pose = np.array([0.5189679577866968, -0.4026275362682761, 0.3352242119492667,
                           0.004476505618505988, 0.7008507652632687, 0.7131028200749294, -0.016219465639604885])
    
    # # homing
    while 1:
        lfd.go_to_start_pose(start_pose, stage='homing')
        ans = input('Repeat? [y/n]: ')
        if ans=='n':
            break
    
    # predict
    lfd.run_model()

