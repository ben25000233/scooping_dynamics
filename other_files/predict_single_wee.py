import rospy
from sensor_msgs.msg import Image as msg_Image
from cv_bridge import CvBridge
import numpy as np
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from pynput.keyboard import Listener, KeyCode
import math
import dynamic_reconfigure.client
import os
import roslaunch
import cv2
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import time
import quaternion
from model import DiffusionPolicy, EMAModel, RotationTransformer, SingleObEncoder
# from torch_ema import ExponentialMovingAverage
import copy
import torchvision.models as models
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from pyconfigparser import configparser
import open3d as o3d



class LfD():
    def __init__(self):

        # --------#
        # setting #
        #---------#
        self.control_rate = rospy.Rate(10)
        self.demo_length = 250 # 25 sec

        # ----------#
        # Publisher #
        #-----------#
        self.ee_pub = rospy.Publisher('/equilibrium_pose', PoseStamped, queue_size=0)

        # -----------#
        # Subscriber #
        #------------#

        # joint state
        self.joints_sub = rospy.Subscriber("/joint_states", JointState, self.joint_callback)
        # ee_pose
        self.pos_sub = rospy.Subscriber("/cartesian_pose", PoseStamped, self.ee_pos_callback)
        # f/t sensor
        self.ftsensor = rospy.Subscriber("/force_torque_ext", WrenchStamped, self.ft_callback)
        # camera
        self.rgb_front = rospy.Subscriber("/cam_front/color/image_raw", msg_Image, self.rgb_front_callback)
        self.depth_front = rospy.Subscriber("/cam_front/aligned_depth_to_color/image_raw", msg_Image, self.depth_front_callback)
        self.rgb_ee = rospy.Subscriber("/cam_ee/color/image_raw", msg_Image, self.rgb_ee_callback)
        self.depth_ee = rospy.Subscriber("/cam_ee/aligned_depth_to_color/image_raw", msg_Image, self.depth_ee_callback)
        self.rgb_demo = rospy.Subscriber("/cam_demo/color/image_raw", msg_Image, self.rgb_demo_callback)
        self.depth_demo = rospy.Subscriber("/cam_demo/aligned_depth_to_color/image_raw", msg_Image, self.depth_demo_callback)

        # ---------------#
        # Initialization #
        #--------------- #  
        self.curr_joint = np.array([0., 0., 0., 0., 0., 0., 0.])
        self.curr_pos = np.array([0.5, 0, 0.5])
        self.curr_ori = np.array([1, 0, 0, 0])

        # origin(480,640,3)
        self.rgb_front_image = np.zeros((480, 640, 3))
        self.depth_front_image = np.zeros((480, 640))
        self.rgb_ee_image = np.zeros((480, 640, 3))
        self.depth_ee_image = np.zeros((480, 640))
        self.rgb_demo = np.zeros((480, 640, 3))
        self.depth_demo = np.zeros((480, 640))

        self.init_force_mat = np.zeros((16, 3))
        self.ft_f = np.array([0., 0., 0.])
        self.ft_t = np.array([0., 0., 0.])
        self.step_change = 0.1
        
        self.recorded_traj = None
        self.recorded_ori = None

        self.ros_time = 0
        self.file_name = None
        self.listener = Listener(on_press=self._on_press)
        self.listener.start()
        self.end = False
    
    def joint_callback(self, data):
        self.curr_joint = np.array(data.position[0:7])
     
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
    
    def to_eular(self, pose):
        # ori [w,x,y,z]
        orientation_list = [pose[4], pose[5], pose[6], pose[3]]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        return np.array([pose[0], pose[1], pose[2], roll, pitch, yaw])
    
    def to_qua(self, pose):
        # ori [roll,pitch,yaw]
        orientation_list = [pose[3], pose[4], pose[5]]
        q = quaternion_from_euler(orientation_list)
        return np.array([pose[0], pose[1], pose[2], q[0], q[1], q[2], q[3]])
    
    # ------ #
    # Homing #
    #--------#

    def go_to_start_pose(self, goal_pose):
        rospy.wait_for_message("/cartesian_pose", PoseStamped)
        print("Go to start pose.")

        start = self.curr_pos
        start_ori = self.curr_ori

        goal_ = goal_pose[0:3]

        squared_dist = np.sum(np.subtract(start, goal_)**2, axis=0)
        dist = np.sqrt(squared_dist)
        print("dist", dist)
        interp_dist = 0.001  # [m]
        step_num_lin = math.floor(dist / interp_dist)
        
        print("num of steps linear", step_num_lin)
        
        q_start = np.quaternion(start_ori[0], start_ori[1], start_ori[2], start_ori[3])
        print("q_start", q_start)
        q_goal = np.quaternion(goal_pose[3], goal_pose[4], goal_pose[5], goal_pose[6])

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
        ])(img)

    def depth_transform(self, img):
        return transforms.Compose([
            transforms.Resize([240, 320]),
            transforms.ToTensor(),
        ])(img)
    
    def run_model(self):

        rospy.wait_for_message("/cartesian_pose", PoseStamped)
        rospy.wait_for_message("/cam_front/color/image_raw", msg_Image)
        rospy.wait_for_message("/cam_front/aligned_depth_to_color/image_raw", msg_Image)
        # rospy.wait_for_message("/cam_demo/color/image_raw", msg_Image)
        # rospy.wait_for_message("/cam_demo/aligned_depth_to_color/image_raw", msg_Image)

        self.set_stiffness(2000, 2000, 2000, 50, 50, 50, 0.0)

        cfg = configparser.get_config(file_name='grits_singleview.yaml')  
        rotation_transformer_input = RotationTransformer('quaternion', 'rotation_6d')      
        rotation_transformer_output = RotationTransformer('rotation_6d', 'quaternion')
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
        
        # load 
        checkpoint = torch.load('epoch=1000.pth', map_location=cfg.training.device)
        ema.load_state_dict(checkpoint['ema_state_dict'])

        policy = ema
        policy.eval()

        rospy.sleep(1)

        print("Start to predict.")

        rgb_input = []
        depth_input = []
        ee_input = []

        # rgb_front_save = []
        # depth_front_save = []
        # traj_save = []

        with torch.no_grad():
            step = 0
            while not rospy.is_shutdown():

                if self.end == True : 
                    break
                
                if step<cfg.n_obs_steps-1:
                    rgb_input.append(self.rgb_transform(Image.fromarray(np.uint8(self.rgb_front_image)).convert('RGB')))
                    depth_input.append(self.depth_transform(Image.fromarray(np.uint8(self.depth_front_image), 'L')))
                    ee_rotation6d = rotation_transformer_input.forward(self.curr_ori)
                    ee_input.append(torch.from_numpy(np.concatenate((self.curr_pos, ee_rotation6d), -1)))
                    step += 1
                    continue
                else:
                    # current
                    rgb_input.append(self.rgb_transform(Image.fromarray(np.uint8(self.rgb_front_image)).convert('RGB')))
                    depth_input.append(self.depth_transform(Image.fromarray(np.uint8(self.depth_front_image), 'L')))
                    ee_rotation6d = rotation_transformer_input.forward(self.curr_ori)
                    ee_input.append(torch.from_numpy(np.concatenate((self.curr_pos, ee_rotation6d), -1)))

                    assert len(rgb_input)==cfg.n_obs_steps
                    assert len(depth_input)==cfg.n_obs_steps
                    assert len(ee_input)==cfg.n_obs_steps

                    # save image
                    # rgb_front_save.append(np.uint8(self.rgb_front_image))
                    # depth_front_save.append(np.uint8(self.depth_front_image))
                    
                    rgb_in = torch.stack(rgb_input, dim=0) # [5, 3, 240, 320]
                    depth_in = torch.stack(depth_input, dim=0)  # [2, 1, 240, 320]
                    obs_in = torch.unsqueeze(torch.cat((rgb_in, depth_in), 1), 0).to('cuda:0') # [1, 2, 4, 240. 320]
                    ee_in = torch.unsqueeze(torch.stack(ee_input, dim=0), 0).to('cuda:0', dtype=torch.float32)

                    # predict
                    action, _ = policy.predict_action((obs_in, ee_in)) 
                    # transform rotate
                    action_publish = action.cpu().detach().numpy().squeeze(0) # [8, 9]
                    action_position = action_publish[:, 0:3]
                    action_rotation = rotation_transformer_output.forward(action_publish[:, 3:])
                    action_publish = np.concatenate((action_position, action_rotation), -1) # [8, 7]

                    # save trajectory
                    # traj_save.append(action_publish)
                    
                    # publish
                    for robot_step in range(action_publish.shape[0]):

                        command = action_publish[robot_step, :]
                        goal = PoseStamped()
                        goal.pose.position.x = command[0]
                        goal.pose.position.y = command[1]
                        goal.pose.position.z = command[2]
                        
                        goal.pose.orientation.w = command[3]
                        goal.pose.orientation.x = command[4]
                        goal.pose.orientation.y = command[5]
                        goal.pose.orientation.z = command[6]

                        print("i: ", step)
                        if np.max(abs(command[0:3] - np.array(self.curr_pos)))>0.05:
                            self.go_to_start_pose(command)
                        else:
                            self.ee_pub.publish(goal)

                        self.control_rate.sleep()
                        step += 1

                        # add to input list
                        rgb_input.append(self.rgb_transform(Image.fromarray(np.uint8(self.rgb_front_image)).convert('RGB')))
                        depth_input.append(self.depth_transform(Image.fromarray(np.uint8(self.depth_front_image), 'L')))
                        ee_rotation6d = rotation_transformer_input.forward(self.curr_ori)
                        ee_input.append(torch.from_numpy(np.concatenate((self.curr_pos, ee_rotation6d), -1))) 

                        # save image
                        # rgb_front_save.append(np.uint8(self.rgb_front_image))
                        # depth_front_save.append(np.uint8(self.depth_front_image))


                for _ in range(cfg.n_action_steps+1):
                  rgb_input.pop(0)
                  depth_input.pop(0)
                  ee_input.pop(0)

                assert len(rgb_input)==cfg.n_obs_steps-1
                assert len(depth_input)==cfg.n_obs_steps-1     
                assert len(ee_input)==cfg.n_obs_steps-1           

            print("Finish task.")

            # save all
            # food_name = input('Save name: ')
            # if not os.path.exists('demo/' + food_name):
            #     os.mkdir('demo/' + food_name)
            # np.save('demo/' + food_name + '/' + 'rgb_front.npy', np.array(rgb_front_save))
            # np.save('demo/' + food_name + '/' + 'depth_front.npy', np.array(depth_front_save))
            # np.save('demo/' + food_name + '/' + 'traj.npy', np.array(traj_save))
            # break
        

if __name__ == '__main__':

    rospy.init_node('run_model', anonymous=True) 
    rospy.loginfo('started')
    rospy.sleep(1)
    input("press enter to start.")

    lfd = LfD()
    start_pose = np.array([0.5189679577866968, -0.4026275362682761, 0.3352242119492667,
                           0.004476505618505988, 0.7008507652632687, 0.7131028200749294, -0.016219465639604885])
    
    # homing
    while 1:
        lfd.go_to_start_pose(start_pose)
        ans = input('Repeat? [y/n]: ')
        if ans=='n':
            break
    
    # predict
    lfd.run_model()