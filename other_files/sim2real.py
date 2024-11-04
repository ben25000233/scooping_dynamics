from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *

import os
import yaml
import torch
import numpy as np
import open3d as o3d
import random
import h5py

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from BallGenerator import BallGenerator
import time

class IsaacSim():
    def __init__(self):
        self.grain_type = "solid"
        
        self.default_height = 0.2
        #tool_type : spoon, knife, stir, fork
        self.tool = "spoon"

        self.config_file = "./dynamics/collect_time.yaml"

        with open(self.config_file, 'r') as file:
            self.config = yaml.safe_load(file)

        self.count = int(self.config['count'])


        count = int(self.count) + 1
        self.config['count'] = count
        #self.config['count'] = 0

        with open(self.config_file, 'w') as file:
            yaml.safe_dump(self.config, file)
        
        
        # initialize gym
        self.gym = gymapi.acquire_gym()
        #self.domainInfo = WeighingDomainInfo(domain_range=None, flag_list=None)

        # create simulator
        self.env_spacing = 1.5
        self.max_episode_length = 195
        self.asset_root = "urdf"
        self.gravity = -9.8
        self.create_sim()


        
        # create viewer using the default camera properties
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        # Look at the first env
        self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, self.cam_target)


        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_dofs, 2)

        
        self.All_poses = [torch.tensor([], device=self.device) for _ in range(self.num_envs)]
        
        
        self.All_steps = np.zeros(self.num_envs)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        _rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state_tensor = gymtorch.wrap_tensor(_rb_state_tensor).view(-1, 13)

        self.action_space = {
            "up": torch.Tensor([[0,  -0.1,  0,  0.3,  0., -0.15, 0.]]),
            "down": torch.Tensor([[0, 0.1, 0, -0.3, 0,  0.2,  0.]]),
            "left": torch.Tensor([[-0.0229, -0.0040, -0.0490, -0.0030, -0.0314,  0.0031, -0.0545]]),
            "right": torch.Tensor([[0.0219,  0.0136,  0.0486,  0.0114,  0.0303, -0.0003,  0.0544]]),
            "forward": torch.Tensor([[0.5144, -0.0957,  0.0610, -0.0206,  0.2810, -0.1702,  0.4203]]),
            #"forward": torch.Tensor([[0,  0.1, 0,  0.15, 0,  0.01, 0]]),

            "scoop_up": torch.Tensor([[0, -0.4,  0, -0.45,  0.,  0.45,  0]]),    

            "test":  torch.Tensor([[0, 0,  0, 0,  0.,  0.1,  0]]),
            "backward": torch.Tensor([[0, -0.1,  0.003, -0.06,  0.003, -0.04,  0]]),
            "scoop_down": torch.Tensor([[ 0, 0,  0, -0.1,  0, -0.1,  0]]),
            "rest" : torch.Tensor([[0,0,0,0,0,0,0]])
        }
        

    def create_sim(self):
        
        # parse arguments
        args = gymutil.parse_arguments()
        args.physics_engine = gymapi.SIM_FLEX
        args.use_gpu = True
        args.use_gpu_pipeline = False
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        self.num_envs = 1
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

        
        if self.grain_type == "solid":
            sim_params.dt = 1.0/60
            sim_params.substeps = 3
        else : 
            sim_params.dt = 1.0/60
            sim_params.substeps = 3
 
        sim_params.flex.solver_type = 5
        sim_params.flex.relaxation = 1
        sim_params.flex.warm_start = 0.75
        sim_params.flex.friction_mode = 2

        sim_params.flex.shape_collision_margin = 0.0001
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 20

        sim_params.flex.max_soft_contacts = 600000

        
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)


        self._create_ground_plane()
        self._create_envs(self.num_envs, self.env_spacing, int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def create_table(self):

        # Load Bowl asset
        file_name = 'table/table.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.fix_base_link = True
        self.table_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        
    def create_bowl(self):

        # Load Bowl asset
        file_name = 'bowl/bowl.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.fix_base_link = True
        asset_options.vhacd_params.resolution = 1000000
        self.bowl_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)

        # file_name = 'bowl/transparant_bowl.urdf'
        # asset_options = gymapi.AssetOptions()
        # asset_options.armature = 0.01
        # asset_options.vhacd_enabled = True
        # asset_options.fix_base_link = True
        # self.transparent_bowl_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)

    def create_franka(self):
        # create franka asset
        self.num_dofs = 0
        asset_file_franka = "franka_description/robots/" + self.tool + "_franka.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.disable_gravity = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 1000000
        self.franka_asset = self.gym.load_asset(self.sim, self.asset_root, asset_file_franka, asset_options)
        self.franka_dof_names = self.gym.get_asset_dof_names(self.franka_asset)
        self.num_dofs += self.gym.get_asset_dof_count(self.franka_asset)

        self.hand_joint_index = self.gym.get_asset_joint_dict(self.franka_asset)["panda_hand_joint"]
        self.ee_handles = []
        # set franka dof properties
        self.franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        self.franka_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        self.franka_dof_props["stiffness"][:].fill(3000.0)
        # self.franka_dof_props["armature"][:] = 100
      
        self.franka_dof_props["damping"][:].fill(500.0)
       
   
        self.franka_dof_props["effort"][:] = 500

        # set default pose
        self.franka_start_pose = gymapi.Transform()
        self.franka_start_pose.p = gymapi.Vec3(0, 0.0, 0.0)
        self.franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        asset_options.tendon_limit_stiffness = 4000
        

    def add_franka(self):
        # create franka and set properties
        self.franka_handle = self.gym.create_actor(self.env_ptr, self.franka_asset, self.franka_start_pose, "franka", 0, 4, 2)
       
        body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.franka_handle)
        for k in range(11):
            body_shape_prop[k].thickness = 0.001
            body_shape_prop[k].friction = 0
     
            
        self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.franka_handle, body_shape_prop)

        franka_sim_index = self.gym.get_actor_index(self.env_ptr, self.franka_handle, gymapi.DOMAIN_SIM)
        self.franka_indices.append(franka_sim_index)

        # self.franka_dof_index = [
        #     self.gym.find_actor_dof_index(self.env_ptr, self.franka_handle, dof_name, gymapi.DOMAIN_SIM)
        #     for dof_name in self.franka_dof_names
        # ]
        self.franka_dof_index = [0,1,2,3,4,5,6,7,8]

        franka_hand = self.gym.find_actor_rigid_body_handle(self.env_ptr, self.franka_handle, "panda_hand")
        self.ee_handles.append(franka_hand)
        self.franka_hand_sim_idx = self.gym.find_actor_rigid_body_index(self.env_ptr, self.franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
        self.gym.set_actor_dof_properties(self.env_ptr, self.franka_handle, self.franka_dof_props)

        
    def create_bolt(self):

        # Load bolt asset
        file_name = 'grains/bolt.urdf'
        asset_options = gymapi.AssetOptions()
        self.between_ball_space = 0.1
        asset_options.armature = 0.01
        asset_options.vhacd_params.resolution = 500000
        self.bolt_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        

    def create_ball(self):
        self.ball_radius = round(random.uniform(0.003, 0.006), 3)
        self.ball_mass = round(random.uniform(0.002, 0.01), 3)
        self.ball_friction = round(random.uniform(0, 0.3),2)
        max_num = int(60/pow(2, (self.ball_radius - 0.003)*1000))
        self.ball_amount = random.randint(int(max_num/3), max_num)
        self.ball_amount = 40


        self.between_ball_space = self.ball_radius*10
        ballGenerator = BallGenerator()
        file_name = 'BallHLS.urdf'
        ballGenerator.generate(file_name=file_name, ball_radius=self.ball_radius, ball_mass=self.ball_mass, type = "solid")
        self.ball_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, gymapi.AssetOptions())

    def set_ball_property(self, ball_pose):
        
        ball_handle = self.gym.create_actor(self.env_ptr, self.ball_asset, ball_pose, "grain", 0, 0)
        body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, ball_handle)

        body_shape_prop[0].friction = self.ball_friction
        body_shape_prop[0].contact_offset = 0.0001   # Distance at which contacts are generated
        body_shape_prop[0].rest_offset = 0      # How far objects should come to rest from the surface of this body 
        body_shape_prop[0].restitution = 0.1     # when two objects hit or collide, the speed at which they move after the collision
        body_shape_prop[0].thickness = 0       # the ratio of the final to initial velocity after the rigid body collides. 
        
        self.gym.set_actor_rigid_shape_properties(self.env_ptr, ball_handle, body_shape_prop)
        c = np.array([115, 78, 48]) / 255.0
        color = gymapi.Vec3(c[0], c[1], c[2])
        color = gymapi.Vec3(1, 1, 1)
        self.gym.set_rigid_body_color(self.env_ptr, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        
        return ball_handle
    


    def add_solid(self):
        #add balls
        ball_amount = self.ball_amount
        ball_amount = 10
        
        ball_pose = gymapi.Transform()
        z = z = 0.1 + self.ball_radius
        ball_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.ball_handle = []
        ran = 8
    
        while ball_amount > 0:
            y = -0.2
            for j in range(ran):
                x = 0.49
                for k in range(ran):
                    ball_pose.p = gymapi.Vec3(x, y, z)
                    ball_handle = self.set_ball_property(ball_pose)
                    self.ball_handle.append(ball_handle)
                    x += self.ball_radius*2 
                y += self.ball_radius*2 
            z += self.ball_radius*2
            ball_amount -= 1
  

        

    
    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, 0.75 * -spacing, 0.0)
        upper = gymapi.Vec3(0, 0.15 * spacing, spacing)
        self.create_bowl()
        self.create_ball()
        self.create_table()
        self.create_franka()
        self.create_bolt()
        
    
        # cache some common handles for later use
        self.camera_handles = [[]for _ in range(self.num_envs)]
        self.franka_indices = []
        self.envs = []
        self.ee_handles = []

        #set camera
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 640
        self.camera_props.height = 480
        self.camera_props.horizontal_fov = 69
        
        #store ball info
        self.spillage_amount = np.zeros(self.num_envs)
        self.pre_spillage = np.zeros(self.num_envs)

        self.ball_handles = [[] for _ in range(self.num_envs)]
        self.spillage_amount = [[] for _ in range(self.num_envs)]
        self.record_ee_pose = [[] for _ in range(self.num_envs)]
        self.hand_pcd_list = [[] for _ in range(self.num_envs)]
        self.front_pcd_list = [[] for _ in range(self.num_envs)]

        # create and populate the environments
        for i in range(num_envs):
            # create env
            self.env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(self.env_ptr)

            # add bowl_1
            bowl_pose = gymapi.Transform()
            bowl_pose.r = gymapi.Quat(0, 0, 0, 1)
            bowl_pose.p = gymapi.Vec3(0.523, -0.17 , 0.0575)   
            self.bowl_1 = self.gym.create_actor(self.env_ptr, self.bowl_asset, bowl_pose, "bowl_1", 0, 0)

            body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.bowl_1)
            # thickness(soft) = 0.0003, thickness(soft) = 0.007
            body_shape_prop[0].thickness = 0.0005      # the ratio of the final to initial velocity after the rigid body collides.(but have no idea why it will affect the contact distance) 
            body_shape_prop[0].friction = 0.1
            self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.bowl_1, body_shape_prop)
            
            
            # #add bowl_2
            # bowl_pose = gymapi.Transform()
            # bowl_pose.r = gymapi.Quat(0, 0, 0, 1)
            # bowl_pose.p = gymapi.Vec3(0.7, -0.5 , self.default_height/2)   
            # self.bowl_2 = self.gym.create_actor(self.env_ptr, self.transparent_bowl_asset, bowl_pose, "bowl_2", 0, 0)

            # body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.bowl)
            # body_shape_prop[0].thickness = 0.005
            # self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.bowl, body_shape_prop)
        
            # add tabel
            # table_pose = gymapi.Transform()
            # table_pose.r = gymapi.Quat(0, 0, 0, 1)
            # table_pose.p = gymapi.Vec3(0.5, -0.5 , -0.22)   
            # self.table = self.gym.create_actor(self.env_ptr, self.table_asset, table_pose, "table", 0, 0)
            # color = gymapi.Vec3(144/255, 164/255, 203/255)
            # self.gym.set_rigid_body_color(self.env_ptr, self.table, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            
            
            # body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.table)
            # body_shape_prop[0].thickness = 0.0005      
            # body_shape_prop[0].friction = 0.5
            # self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.table, body_shape_prop)
            
            # add ball
            
            self.add_solid()
            # self.ball_handles[i] = self.ball_handle



            #add franka
            self.add_franka()

            #add camera_1
            #data = np.load('cam2base.npy')
    

            self.cam_pos = gymapi.Vec3(0.6, -0.5, 0.3)
            self.cam_pos = gymapi.Vec3(0.5, 0.7 , 0.3)
            # self.cam_pos = gymapi.Vec3(1.04411428, -0.01324678, 0.81564328)
            #self.cam_target = gymapi.Vec3(0.5, 0.0, self.default_height/2)
            self.cam_target = gymapi.Vec3(0.6, 0.0, 0.0)

            camera_1 = self.gym.create_camera_sensor(self.env_ptr, self.camera_props)
            self.gym.set_camera_location(camera_1, self.env_ptr, self.cam_pos, self.cam_target)
            self.camera_handles[i].append(camera_1)

            # add camera_2(need modify)
            camera_2 = self.gym.create_camera_sensor(self.env_ptr, self.camera_props)
            camera_offset = gymapi.Vec3(0.032, -0.0345, 0.094)
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(275.8))
          
            self.gym.attach_camera_to_body(camera_2, self.env_ptr, self.ee_handles[i], gymapi.Transform(camera_offset, camera_rotation),
                                    gymapi.FOLLOW_TRANSFORM)
            self.camera_handles[i].append(camera_2)

        self.franka_actor_indices = to_torch(self.franka_indices, dtype = torch.int32, device = "cpu")


    def cal_spillages(self, env_index, reset = 0):
        
        spillage_amount = 0
        for ball in self.ball_handles[env_index]:
            body_states = self.gym.get_actor_rigid_body_states(self.envs[env_index], ball, gymapi.STATE_ALL)
            z = body_states['pose']['p'][0][2]

            if z < 0.1:
                spillage_amount += 1

        if reset == 0:
            #print(f"spillage : {int(spillage_amount - self.pre_spillage[env_index])}")
            # if int(spillage_amount - self.pre_spillage[env_index]) > 0 :
            #     self.spillage_amount[env_index].append(1.0)
            # else : 
            #     self.spillage_amount[env_index].append(0.0)
            self.spillage_amount[env_index].append(int(spillage_amount - self.pre_spillage[env_index]))
       
        else : 
            self.pre_spillage[env_index] = int(spillage_amount)

    


    def move_generate(self, franka_idx):

   
        if self.round[franka_idx] %4 == 0:
            first_down = random.randint(min((self.round[franka_idx]+15), 30), 30)
            rest_num = 50

        else : 
            first_down = 0
            rest_num = 10

        forward_num = random.randint(0, 30)
        L_num = random.randint(0, 30)
        R_num = random.randint(0, 30)
        scoop_num = 28
        
        action_list =  ["left"] * L_num + ["right"]*R_num + ["scoop_up"] * scoop_num + ["forward"] * forward_num 
        

        # Shuffle the list randomly
        random.shuffle(action_list)
        # first "rest" for waiting the init grain setup and get the correct scene image
        # last "rest" for waiting the balls drops to calculate the spillage amount
        action_list = ["rest"] * rest_num + ["down"] * first_down  + action_list + ["rest"] * 10 

        self.delta = 0.3
    
        dposes = torch.cat([self.action_space.get(action) for action in action_list])

 

        self.All_poses[franka_idx] = dposes
        self.All_steps[franka_idx] = len(dposes)
        
    
    def reset_franka(self, init_pose):
        action = init_pose
        action = np.append(action, 0.02)
        action = np.append(action, 0.02)
        franka_init_pose = torch.tensor([action], dtype=torch.float32)
        

        
        self.dof_state[:, self.franka_dof_index, 0] = franka_init_pose
        self.dof_state[:, self.franka_dof_index, 1] = 0
       
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(self.franka_actor_indices),
            len(self.franka_actor_indices)
        )




    def data_collection(self, action_list):

        self.collect_time = 10
        action = "rest"
        self.round = [0]*self.num_envs
        self.pos_action = torch.zeros((self.num_envs, 9), dtype=torch.float32)

        self.frame = 0
        dpose_index =  np.zeros(self.num_envs ,dtype=int)
       
        for i in range(self.num_envs):
            self.move_generate(i)

        action_index = 0
        self.check = 0
        

        while not self.gym.query_viewer_has_closed(self.viewer):
            
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)


            if self.frame == 0 :
                self.reset_franka(action_list[0])

            if self.frame > 0 : 
 
              
            
                if self.frame % 6 == 0:
                    # print(action_index)
                    # eepose = np.load("./dataset3/cb01/ee_pose_qua.npy")
                    # # print(f"gd_eepose : {eepose[action_index]}")
                    # ee_pose = self.gym.get_rigid_transform(self.envs[0], self.ee_handles[0])
                    # ee_pose_arr  = np.array([ee_pose.p.x, ee_pose.p.y, ee_pose.p.z - 0.1, ee_pose.r.w, ee_pose.r.x, ee_pose.r.y, ee_pose.r.z])
                    # print(f"cal_ee : {ee_pose_arr}")

                    # distance = np.linalg.norm(eepose[action_index] - ee_pose_arr)
                    # print(f"distance : {distance}")
         
                    action_index += 1
                    self.get_pcd(0)
               

                if action_index == action_list.shape[0]:
                    self.reset_franka(action_list[0])
                    action_index = 1
                    exit()

                
                
                action = action_list[0]
                
                
                #print(action_list[action_index+1] - action_list[action_index])
                action = np.append(action, 0.02)
                action = np.append(action, 0.02)

                action = torch.tensor([action], dtype=torch.float32)
               
            
                self.gym.set_dof_position_target_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(action),
                    gymtorch.unwrap_tensor(self.franka_actor_indices),
                    len(self.franka_actor_indices)
                )
       
                if action_index == 130:
                    self.check = 1
                
                    
                 


            # if dpose_index[i] == 20 :
            #     self.cal_spillages(i, reset = 1) 
            # if self.All_steps[i] == 0:
            #     self.cal_spillages(i, reset = 0)
   


            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
            self.frame += 1

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def list_to_nparray(self, lists, type = None):
        temp_array = []
        if type == "next_eepose":
            for i in range(len(lists)):
                temp_array.append(np.array(lists[i][1:self.collect_time*4+1]))
        else:
            for i in range(len(lists)):
                # print(np.array(lists[i][:self.collect_time*4]).shape)
                temp_array.append(np.array(lists[i][:self.collect_time*4]))
        temp = np.stack(temp_array, axis=0)

        shape = temp.shape
        new_shape = (shape[0] * shape[1],) + shape[2:]
        temp_1 = temp.reshape(new_shape )
 
        return temp_1
            

    def get_pcd(self, env_index):

        # get camera images
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
    
        
        if self.round[env_index] < (self.collect_time*4) :
            
            #get front_pcd
            
            front_depth_image = self.gym.get_camera_image(self.sim, self.envs[env_index], self.camera_handles[env_index][0],  gymapi.IMAGE_DEPTH)
            front_color_image = self.gym.get_camera_image(self.sim, self.envs[env_index], self.camera_handles[env_index][0],  gymapi.IMAGE_COLOR)
            plt.imshow(front_depth_image)
            plt.show()
            exit()
            # plt.savefig('cb_depth.jpg')
            # point_cloud = self.depth_to_point_cloud(env_index, front_depth_image, type = 0)
            # self.front_pcd_list[env_index].append(np.array(point_cloud.points))

            
            #get hand_pcd

            hand_depth_image = self.gym.get_camera_image(self.sim, self.envs[env_index], self.camera_handles[env_index][1],  gymapi.IMAGE_DEPTH)
            hand_color_image = self.gym.get_camera_image(self.sim, self.envs[env_index], self.camera_handles[env_index][1],  gymapi.IMAGE_COLOR)
            hand_seg_image = self.gym.get_camera_image(self.sim, self.envs[env_index], self.camera_handles[env_index][1],  gymapi.IMAGE_SEGMENTATION)
            hand_color_image = hand_color_image.reshape(self.camera_props.height, self.camera_props.width, 4)
            # plt.imshow(hand_depth_image)
            # plt.savefig('cb_depth.jpg')
            # plt.imshow(hand_color_image)
            # plt.savefig('cb_color.jpg')
            # plt.imshow(hand_seg_image)
            # plt.savefig('cb_seg.jpg')
            # exit()

            
            
            # point_cloud = self.depth_to_point_cloud(env_index, hand_depth_image, type = 1)
            # self.hand_pcd_list[env_index].append(np.array(point_cloud.points))

            if(self.check == 1):
                rotated_image = np.rot90(hand_color_image, k=2)
                plt.imshow(hand_color_image)
                plt.show()
                exit()
            
    
            
            # o3d.visualization.draw_geometries([point_cloud])
  
            


    def depth_to_point_cloud(self, env_index, depth_image, type = 0):
  
        # plt.figure()
        # plt.imshow(depth_image)
        # plt.axis('off')
        # plt.show()
    
        points = []
        vinv = np.linalg.inv(np.matrix(self.gym.get_camera_view_matrix(self.sim, self.envs[env_index], self.camera_handles[env_index][type])))

        proj = self.gym.get_camera_proj_matrix(self.sim, self.envs[env_index], self.camera_handles[env_index][type])
        fu = 2/proj[0, 0]
        fv = 2/proj[1, 1]


        centerU = self.camera_props.width/2
        centerV = self.camera_props.height/2
        for i in range(self.camera_props.width):
            for j in range(self.camera_props.height):
            
                # if depth_image[j, i] > -0.3:
                u = -(i-centerU)/(self.camera_props.width)  # image-space coordinate
                v = (j-centerV)/(self.camera_props.height)  # image-space coordinate
                d = depth_image[j, i]  # depth buffer value
                X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                p2 = X2*vinv  # Inverse camera view to get world coordinates
                points.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                    
        points_np = np.array(points)

        # Create an Open3D PointCloud object
        point_cloud = o3d.geometry.PointCloud()


        # Assign points to the point cloud
        point_cloud.points = o3d.utility.Vector3dVector(points_np)
        
        remaining_cloud = self.align_point_cloud(point_cloud)
         

        # Visualize the point cloud
        o3d.visualization.draw_geometries([point_cloud])
    
        return remaining_cloud
    

    def align_point_cloud(self, pcd, target_points=2000):
        

        plane_model, inliers = pcd.segment_plane(distance_threshold=10,
                                         ransac_n=3,
                                         num_iterations=1000)

        # plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
        #                                  ransac_n=3,
        #                                  num_iterations=1000)
        
        remaining_cloud = pcd.select_by_index(inliers, invert=True)

        num_points = len(remaining_cloud.points)

        target_points = int(num_points)


        if num_points >= target_points:
            # Randomly downsample to target_points
            indices = np.random.choice(num_points, target_points, replace=False)
            new_pcd = remaining_cloud.select_by_index(indices)

        else:
            # Upsample to target_points
            additional_points_needed = target_points - num_points
            existing_points = np.asarray(remaining_cloud.points)
            
            # Repeat points to get the required number of points
            additional_indices = np.random.choice(num_points, additional_points_needed, replace=True)
            additional_points = existing_points[additional_indices]
            
            # Combine original points with additional points
            new_points = np.vstack((existing_points, additional_points))
            new_pcd.points = o3d.utility.Vector3dVector(new_points)
    
        return new_pcd
        
    

    def keyboard_control(self):
        # keyboard event
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_DOWN, "down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "right")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "backward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "forward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "scoop_up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "scoop_down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "test")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_X, "save")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "quit")


    def show_depth(self, images):
 
        for i in range(0, images.shape[0], 10):
            pcd = self.depth_to_point_cloud(0, images[i], type = 0)
    
    def read_pcd(self, pcd):
        pcd = self.align_point_cloud(pcd)
        # Visualize the point cloud
        points = np.asarray(pcd.points)

        # Compute the heights (z-values)
        z_values = points[:, 1]

        # Normalize the z-values to range [0, 1]
        z_min = z_values.min()
        z_max = z_values.max()
        z_normalized = (z_values - z_min) / (z_max - z_min)

        # Create a colormap based on the normalized height
        colors = plt.cm.viridis(z_normalized)[:, :3]  # Using viridis colormap, exclude the alpha channel

        # Assign the colors to the point cloud
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualize the colored point cloud
        o3d.visualization.draw_geometries([pcd])
        

def normalize_image(img, type):
    # if type == "hand" :
    #     image = img[25:150,35:150]
    # elif type == "front" : 
    #     image = img[30:135, 45:165]
    image = img
    denoise_image = image[(image > 0)]
    
    min_val = np.min(denoise_image)
    max_val = np.max(denoise_image)

    if type == "front" :
        min_val += 400

    image[image <= min_val] = min_val

    # Normalize the image to 0-1 range
    normalized_img = (image - min_val) / (max_val - min_val)

    
    return normalized_img


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # create_gif(image_list, "spillage_h.gif")
    
    from PIL import Image
    # issac = IsaacSim()
    # joint = np.load("./cb01/joint_states.npy")
    # issac.data_collection(joint[30:32])
   
    # ply_path = "xyz.ply"
    # point_cloud = o3d.io.read_point_cloud(ply_path)
    # issac.read_pcd(point_cloud)




    # depth_list = np.load("./dataset3/brh03/depth_ee.npy")
    # issac.show_depth(depth_list)   
    import cv2

    image_list = np.load("./sym01/depth_front.npy")
    for i in range(80, 200):
        image = image_list[i]
        resized_img = cv2.resize(image, (320, 240), interpolation=cv2.INTER_LINEAR)

        # hand : 25:130,35:150
        # front : 30:135, 50:165
        img = resized_img[30:135, 50:165]
        nor_img = normalize_image(img, "front")
        
        plt.imshow(nor_img)
        # plt.axis('off')
        # plt.savefig('front.jpg')
        plt.show()
        exit()
    

    # plt.savefig('real.jpg')
    # exit()

   
    # min_val = np.min(depth_image)
    # max_val = np.max(depth_image)

    # # Normalize the image to the range [0, 1]
    # normalized_depth_image = (depth_image - min_val) / (max_val - min_val)
    # plt.imshow(normalized_depth_image)
    # # plt.savefig('cb_depth.jpg')
    # plt.axis('off')
    # plt.show()
    # exit()

    # image_list = np.load("./sym01/depth_ee.npy")
    
    # for idx in range(80, 165):
    #     plt.imshow(image_list[idx])
    #     plt.show()

    # file_path = './dynamics/collected_data/dataset/time_0.h5'
    # with h5py.File(file_path, 'r') as h5_file:
    #     # Access a specific dataset
    #     dataset = h5_file['spillage_vol']
    #     print(len(dataset))
        
    #     image_list = dataset[:]
    # for idx in range(0, len(image_list)):
    #     plt.imshow(image_list[idx])
    #     plt.show()


    # sample_trial = np.load("./sample_trial/joint_states.npy")
    # mean_trial = np.load("./sample_trial/mean_trial.npy")
    # np.save("./sample_trial/down_trial.npy", mean_trial[80:115])
    # np.save("./sample_trial/scoop_trial.npy", mean_trial[115:176])
    # exit()
    # np.save("./sample_trial/trans_trial.npy", mean_trial[165:210])


    # down_trial = np.load("./sample_trial/down_trial.npy")
    # scoop_trial = np.load("./sample_trial/scoop_trial.npy")
    # trans_trial = np.load("./sample_trial/trans_trial.npy")

    # mean_trial = np.load("./sample_trial/mean_trial.npy")
    


    
    
