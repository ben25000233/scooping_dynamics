from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *

import os
import time
import yaml
import torch
import numpy as np
import open3d as o3d

import json

import random


from BallGenerator import BallGenerator


class IsaacSim():
    def __init__(self):
        self.grain_type = "solid"
        
        self.default_height = 0.2
        #tool_type : spoon, knife, stir, fork
        self.tool = "spoon"


        # initialize gym
        self.gym = gymapi.acquire_gym()

        # create simulator
        self.env_spacing = 1.5
        self.max_episode_length = 195
        self.asset_root = "urdf"
        self.gravity = -9.8
        self.create_sim()

        # create viewer using the default camera properties
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        # Look at the first env
        self.cam_pos = gymapi.Vec3(0.8, -0.5, 0.7)
        self.cam_target = gymapi.Vec3(0.5, -0.5, self.default_height/2)
        self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, self.cam_target)


        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_dofs, 2)

        self.record_dof = [torch.tensor([], device=self.device) for _ in range(self.num_envs)]
        self.All_poses = [torch.tensor([], device=self.device) for _ in range(self.num_envs)]
        
        self.All_steps = np.zeros(self.num_envs)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        _rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state_tensor = gymtorch.wrap_tensor(_rb_state_tensor).view(-1, 13)

        self.action_space = {
            "up": torch.Tensor([[[0.],[0.],[1.],[0.],[0.],[0.]]]).to(self.device),
            "down": torch.Tensor([[[0.],[0.],[-1.],[0.],[0.],[0.]]]).to(self.device),
            "left": torch.Tensor([[[0.],[-1.],[0.],[0.],[0.],[0.]]]).to(self.device),
            "right": torch.Tensor([[[0.],[1.],[0.],[0.],[0.],[0.]]]).to(self.device),
            "forward": torch.Tensor([[[1.],[0.],[0.],[0.],[0.],[0.]]]).to(self.device),

            "scoop_up": torch.Tensor([[[0.],[0.],[0.],[0.],[10.],[0.]]]).to(self.device),

            "backward": torch.Tensor([[[-1.],[0.],[0.],[0.],[0.],[0.]]]).to(self.device),
            "scoop_down": torch.Tensor([[[0.],[0.],[0.],[0.],[-10.],[0.]]]).to(self.device),
            "rest" : torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]]).to(self.device)
        }

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.j_eef = jacobian[:, self.hand_joint_index, :, :7].to(self.device)
        

    def create_sim(self):
        
        # parse arguments
        args = gymutil.parse_arguments()
        #args.physics_engine = gymapi.SIM_FLEX
        args.use_gpu = True
        args.use_gpu_pipeline = False
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_envs = 2
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, self.gravity)

        
        sim_params.dt = 1.0/60
        sim_params.substeps = 1

 
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1

        sim_params.physx.friction_offset_threshold = 0.0001
        sim_params.physx.friction_correlation_distance = 10
        sim_params.physx.contact_offset = 1
        sim_params.physx.rest_offset = 0.0001
        sim_params.physx.max_depenetration_velocity = 1000

        
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
        file_name = 'bowl/real_bowl.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.fix_base_link = True
        self.bowl_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)

        file_name = 'bowl/trans_real_bowl.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.fix_base_link = True
        self.trans_bowl_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)

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

        # set franka dof properties
        self.franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        self.franka_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        self.franka_dof_props["stiffness"][:].fill(4000.0)
        self.franka_dof_props["damping"][:].fill(1000.0)

        # self.franka_dof_props["effort"] /= 5



        # set default pose
        self.franka_start_pose = gymapi.Transform()
        self.franka_start_pose.p = gymapi.Vec3(0, -0.5, 0.0)
        self.franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)


    def add_franka(self):
        # create franka and set properties
        self.franka_handle = self.gym.create_actor(self.env_ptr, self.franka_asset, self.franka_start_pose, "franka", 0, 4, 2)
       
        body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.franka_handle)
        for k in range(11):
            body_shape_prop[k].thickness = 0.001
            
        self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.franka_handle, body_shape_prop)

        franka_sim_index = self.gym.get_actor_index(self.env_ptr, self.franka_handle, gymapi.DOMAIN_SIM)
        self.franka_indices.append(franka_sim_index)

        # self.franka_dof_index = [
        #     self.gym.find_actor_dof_index(self.env_ptr, self.franka_handle, dof_name, gymapi.DOMAIN_SIM)
        #     for dof_name in self.franka_dof_names
        # ]
        self.franka_dof_index = [0,1,2,3,4,5,6,7,8]

        self.franka_hand = self.gym.find_actor_rigid_body_handle(self.env_ptr, self.franka_handle, "panda_hand")

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

        self.ball_radius = 0.006
        self.ball_mass = 0.01
        self.ball_friction = 0.5
        self.ball_amount = 0


        self.between_ball_space = 0.03
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
        self.gym.set_rigid_body_color(self.env_ptr, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        
        return ball_handle
        
    def add_solid(self):
        #add balls
        ball_amount = self.ball_amount
        ball_pose = gymapi.Transform()
        z = self.default_height/2 +0.03
        ball_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.ball_handle = []
        layer = 0
    
        while ball_amount > 0:
            y = -0.53
            ran = min(ball_amount, 6)
            layer += 1
            for j in range(ran):
                x = 0.48 
                for k in range(ran):
                    ball_pose.p = gymapi.Vec3(x, y, z)
                    ball_handle = self.set_ball_property(ball_pose)
                    self.ball_handle.append(ball_handle)
                    x += self.ball_radius*2 + 0.001*j
                y += self.ball_radius*2 + 0.001*j
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

        #set camera
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1080
        camera_props.height = 720
        
        #store ball info
        self.spillage_amount = np.zeros(self.num_envs)
        self.pre_spillage = np.zeros(self.num_envs)

        self.ball_handles = [[] for _ in range(self.num_envs)]
        self.spillage_amount = [[] for _ in range(self.num_envs)]

        # create and populate the environments
        for i in range(num_envs):
            # create env
            self.env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(self.env_ptr)

            #add bowl_1
            bowl_pose = gymapi.Transform()
            bowl_pose.r = gymapi.Quat(0, 0, 0, 1)
            bowl_pose.p = gymapi.Vec3(0.5, -0.5 , self.default_height/2 )   
            self.bowl_1 = self.gym.create_actor(self.env_ptr, self.bowl_asset, bowl_pose, "bowl_1", 0, 0)

            body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.bowl_1)
            # thickness(soft) = 0.0003, thickness(soft) = 0.007
            body_shape_prop[0].thickness = 0.0005      # the ratio of the final to initial velocity after the rigid body collides.(but have no idea why it will affect the contact distance) 
            body_shape_prop[0].friction = 0
            self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.bowl_1, body_shape_prop)
            
            
            # #add bowl_2
            # bowl_pose = gymapi.Transform()
            # bowl_pose.r = gymapi.Quat(0, 0, 0, 1)
            # bowl_pose.p = gymapi.Vec3(0.7, -0.5 , self.default_height/2)   
            # self.bowl_2 = self.gym.create_actor(self.env_ptr, self.trans_bowl_asset, bowl_pose, "bowl_2", 0, 0)

            # body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.bowl)
            # body_shape_prop[0].thickness = 0.005
            # self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.bowl, body_shape_prop)
        
            # add tabel
            table_pose = gymapi.Transform()
            table_pose.r = gymapi.Quat(0, 0, 0, 1)
            table_pose.p = gymapi.Vec3(0.5, -0.5 , -0.15)   
            self.table = self.gym.create_actor(self.env_ptr, self.table_asset, table_pose, "table", 0, 0)

            # body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.table)
            # body_shape_prop[0].thickness = 0.0005      
            # body_shape_prop[0].friction = 0.5
            # self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.table, body_shape_prop)
            
            #add ball
            if self.grain_type == "solid":
                self.add_solid()
                self.ball_handles[i] = self.ball_handle
                print("amount", len(self.ball_handles[i]))

            
            #add franka
            self.add_franka()

            
            #add camera_1
   
            cam_pos = gymapi.Vec3(0.8, -0.5, 0.7)
            cam_target = gymapi.Vec3(0.5, -0.5, self.default_height/2)
            camera_1 = self.gym.create_camera_sensor(self.env_ptr, camera_props)
            self.gym.set_camera_location(camera_1, self.env_ptr, cam_pos, cam_target)
            self.camera_handles[i].append(camera_1)

            #add camera_2(need modify)
            # camera_2 = self.gym.create_camera_sensor(self.env_ptr, camera_props)
            # camera_offset = gymapi.Vec3(0.1, 0, 0)
            # camera_rotation = gymapi.Quat(0.75, 0, 0.7, 0)
          
            # self.gym.attach_camera_to_body(camera_2, self.env_ptr, self.franka_hand, gymapi.Transform(camera_offset, camera_rotation),
            #                         gymapi.FOLLOW_TRANSFORM)
            # self.camera_handles.append(camera_2)

        self.franka_actor_indices = to_torch(self.franka_indices, dtype = torch.int32, device = "cpu")


    def cal_spillages(self, env_index, init):
    
        spillage_amount = 0
        for ball in self.ball_handles[env_index]:
            body_states = self.gym.get_actor_rigid_body_states(self.envs[env_index], ball, gymapi.STATE_ALL)
            z = body_states['pose']['p'][0][2]

            if z < 0.1:
                spillage_amount += 1

        if init == 1:
            self.spillage_amount[env_index].append(int(spillage_amount - self.pre_spillage[env_index]))
        
        
        self.pre_spillage[env_index] = int(spillage_amount)

    


    def move_generate(self, franka_idx):

        if self.round[franka_idx] %4 == 0:
            first_down = random.randint(min((self.round[franka_idx]*1.5+30 - self.ball_amount), 70), 70)
            rest_num = 50
        else : 
            first_down = 0
            rest_num = 0

    
        down_num = random.randint(30, 60)
        forward_num = random.randint(0, 30)
        L_num = random.randint(0, 30)
        R_num = random.randint(0, 30)
        scoop_num = 40
        
        action_list = ["down"] * down_num + ["left"] * L_num + ["right"]*R_num + ["scoop_up"] * scoop_num + ["forward"] * forward_num
        

        # Shuffle the list randomly
        random.shuffle(action_list)
        # first "rest" for waiting the init grain setup and get the correct scene image
        # last "rest" for waiting the balls drops to calculate the spillage amount
        action_list = ["rest"] * rest_num + ["down"] * first_down  + action_list + ["rest"] * 20 

        
        dposes = torch.cat([self.control_ik(self.action_space.get(action)).squeeze() for action in action_list])
 

        self.All_poses[franka_idx] = dposes
        self.All_steps[franka_idx] = len(dposes)

    def control_ik(self, dpose, damping=0.05):
        print(dpose)
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2).to(self.device)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose

        return u
        
    
    def reset_franka(self, franka_idx):
     
        franka_init_pose = torch.tensor([[-0.0371,  0.0939,  0.0481, -1.8604, -0.0090,  1.5319,  0.8286,  0.02,  0.02]], dtype=torch.float32)
        franka_init_pose = torch.tensor([[-1.0697, -0.1959,  1.0286, -2.1649, -0.6979,  2.0149, -0.5700,  0.02,  0.02]], dtype=torch.float32)
        
        self.dof_state[franka_idx, self.franka_dof_index, 0] = franka_init_pose
        self.dof_state[franka_idx, self.franka_dof_index, 1] = 0

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(self.franka_actor_indices[franka_idx].unsqueeze(0)),
            1
        )



    def data_collection(self):

        if not os.path.exists(f"dynamics/collected_data/top_view/rgb"):
            os.makedirs(f"dynamics/collected_data/top_view/rgb")

        if not os.path.exists(f"dynamics/collected_data/top_view/depth"):
            os.makedirs(f"dynamics/collected_data/top_view/depth")
        # #hand_view
        # if not os.path.exists(f"dynamics/collected_data/hand_view"):
        #     os.makedirs(f"dynamics/collected_data/hand_view")

        
        action = "rest"
        self.round = [0]*self.num_envs
        self.pos_action = torch.zeros((self.num_envs, 9), dtype=torch.float32)

        self.frame = 0
        dpose_index =  np.zeros(self.num_envs ,dtype=int)
       
        for i in range(self.num_envs):
            self.move_generate(i)

        while not self.gym.query_viewer_has_closed(self.viewer):
     
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            
            if self.frame <= 50 : 
                self.frame += 1
                if self.frame == 50 :
                    franka_init_pose = torch.tensor([[-1.0697, -0.1959,  1.0286, -2.1649, -0.6979,  2.0149, -0.5700,  0.02,  0.02]], dtype=torch.float32)

                    self.dof_state[:, self.franka_dof_index, 0] = franka_init_pose
                    self.dof_state[:, self.franka_dof_index, 1] = 0

                    self.gym.set_dof_state_tensor_indexed(
                        self.sim,
                        gymtorch.unwrap_tensor(self.dof_state),
                        gymtorch.unwrap_tensor(self.franka_actor_indices),
                        len(self.franka_actor_indices)
                    )


            else : 
       
                dpose = torch.stack([pose[dpose_index[i]] for i, pose in enumerate(self.All_poses[:])])
            
                self.keyboard_control()
                for evt in self.gym.query_viewer_action_events(self.viewer):
                    action = evt.action if (evt.value) > 0 else "rest"

                dpose = self.control_ik(self.action_space.get(action)).squeeze()
                #print(self.dof_state[:, self.franka_dof_index, 0].squeeze(-1)[:, :7])
                print(dpose)

                dpose_index+=1
                self.pos_action[:, :7] = self.dof_state[:, self.franka_dof_index, 0].squeeze(-1)[:, :7].to(self.device) + dpose
                self.dof_state[:, self.franka_dof_index, 0] = self.pos_action.clone()
                self.All_steps -= 1


            temp = self.dof_state[:, self.franka_dof_index, 0].clone()
            temp[:, -2:] = 0.02

            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(temp),
                gymtorch.unwrap_tensor(self.franka_actor_indices),
                len(self.franka_actor_indices)
            )

            #record dof info
            for i in range(self.num_envs):
                if self.All_steps[i] == 0:
                    
                    self.get_image(i, 1)
                    self.cal_spillages(i, 1)
   
                    
                    self.round[i] += 1
                    self.move_generate(i)
                    dpose_index[i] = 0
                    #self.record_dof[i].append(self.dof_state[i, self.franka_dof_index, 0])

                    if self.round[i] % 4 == 0:
                        self.reset_franka(i)
                        
                #get the init env setting image
                if dpose_index[i] == 60 and self.round[i] % 4 == 0:
          
                    self.get_image(i, 0)
                    self.cal_spillages(i, 0)
               
        
            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def get_image(self, env_index, init = 1):
        '''
        args :
            init = 0 : to get the first image
            init = 1 : to get other part(done) image 
        '''

        # get camera images
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
    
        #get top_image

        self.gym.write_camera_image_to_file(self.sim, self.envs[env_index], self.camera_handles[env_index][0], gymapi.IMAGE_COLOR, f"dynamics/collected_data/top_view/rgb/env_{env_index}_round_{str(int(self.round[env_index]/4))}_{str(init+(self.round[env_index]%4))}.png")
        self.gym.write_camera_image_to_file(self.sim, self.envs[env_index], self.camera_handles[env_index][0], gymapi.IMAGE_DEPTH, f"dynamics/collected_data/top_view/depth/env_{env_index}_round_{str(int(self.round[env_index]/4))}_{str(init+(self.round[env_index]%4))}.png")

        
        #get hand_view
        # self.gym.write_camera_image_to_file(self.sim, self.envs[env_index], self.camera_handles[env_index][1], gymapi.IMAGE_COLOR, f"collected_data/hand_view/rgb_env_{env_index}_round_{str(int(self.round[env_index]/4))}-{str(self.round[env_index]%4)}.png")
        # self.gym.write_camera_image_to_file(self.sim, self.envs[env_index], self.camera_handles[env_index][1], gymapi.IMAGE_DEPTH, f"collected_data/hand_view/depth_env_{env_index}_round_{str(int(self.round[env_index]/4))}-{str(self.round[env_index]%4)}.png")
        

        

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
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "gripper_close")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_X, "save")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "quit")



if __name__ == "__main__":
    issac = IsaacSim()
    issac.data_collection()