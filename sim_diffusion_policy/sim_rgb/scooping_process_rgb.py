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

from BallGenerator import BallGenerator
from scipy.spatial.transform import Rotation as R
from predict_rgb import LfD
import torchvision.transforms as transforms
import cv2
import transforms3d

class IsaacSim():
    def __init__(self):
        self.grain_type = "solid"
        self.lfd = LfD
        self.default_height = 0.2
        #tool_type : spoon, knife, stir, fork
        self.tool = "spoon"
        self.lfd = LfD()
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
        cam_target = gymapi.Vec3(0.5, -0.18, 0.3)

        # Look at the first env
        self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, cam_target)


        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_dofs, 2)

        
        self.All_poses = [torch.tensor([], device=self.device) for _ in range(self.num_envs)]
        
        
        self.All_steps = np.zeros(self.num_envs)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        _rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state_tensor = gymtorch.wrap_tensor(_rb_state_tensor).view(-1, 13)

        self.action_space = {
            "up": torch.Tensor([[[0.],[0.],[1.],[0.],[0.],[0.]]]),
            "down": torch.Tensor([[[0.],[0.],[-1.],[0.],[0.],[0.]]]),
            "rest": torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]]),
            
        }
        
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.j_eef = jacobian[:, self.hand_joint_index, :, :7]
        
    

    def create_sim(self):
        
        # parse arguments
        args = gymutil.parse_arguments()
        #args.physics_engine = gymapi.SIM_FLEX
        args.use_gpu = True
        args.use_gpu_pipeline = False
        self.device = 'cpu'
        self.num_envs = 1
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, self.gravity)

        
        sim_params.dt = 1.0/40
        sim_params.substeps = 15

 
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1

        sim_params.physx.friction_offset_threshold = 0.0001
        sim_params.physx.friction_correlation_distance = 5
        # lead lag
        sim_params.physx.contact_offset = 0.05
        
        sim_params.physx.rest_offset = 0.0001
        sim_params.physx.max_depenetration_velocity = 1

        
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
        asset_options.vhacd_params.resolution = 500000
        self.bowl_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        self.transparent_bowl_asset = self.gym.load_asset(self.sim, self.asset_root, 'bowl/transparent_bowl.urdf', asset_options)

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
        self.ball_radius = round(random.uniform(0.003, 0.008), 4)
        self.ball_mass = round(random.uniform(0.0001, 0.0025), 4)
        self.ball_friction = round(random.uniform(0, 0.2),2)
        max_num = int(32/pow(2, (self.ball_radius - 0.003)*1000))+3
        self.ball_amount = random.randint(1, max_num)
        
        # L : (1, 15) (2,2) (3, 1)
        # M : (1, 20) (2,3) (3, 1)
        # H : (1, 30) (2, 5) (3, 2)
        # ratio : 1-3
        self.radius_ratio = 2
        self.ball_radius = 0.003 * self.radius_ratio
        self.ball_mass = 0.002
        self.ball_friction = 0.1
        self.ball_amount = 2
    
        self.between_ball_space = self.ball_radius*10
        ballGenerator = BallGenerator()
        file_name = 'BallHLS.urdf'
        ballGenerator.generate(file_name=file_name, ball_radius=self.ball_radius, ball_mass=self.ball_mass, type = "solid")
        
        self.ball_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, gymapi.AssetOptions())
    def set_ball_property(self, ball_pose):
        
        ball_handle = self.gym.create_actor(self.env_ptr, self.ball_asset, ball_pose, "grain", 0, 0)
        body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, ball_handle)

        body_shape_prop[0].friction = self.ball_friction
        body_shape_prop[0].contact_offset = 0.001   # Distance at which contacts are generated
        body_shape_prop[0].rest_offset = 0.0003      # How far objects should come to rest from the surface of this body 
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
        ball_pose = gymapi.Transform()
        ball_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.ball_handle = []
        ball_amount = self.ball_amount
    
        z = 0.1 + self.ball_radius
        ran = 4
    
        while ball_amount > 0:
            y = -0.18
            for j in range(ran):
                x = 0.47
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
        self.camera_props.width = 320
        self.camera_props.height = 240
        self.camera_props.horizontal_fov = 57
        
        #store ball info
        self.spillage_amount = np.zeros(self.num_envs)
        self.pre_spillage = np.zeros(self.num_envs)

        self.ball_handles = [[] for _ in range(self.num_envs)]
        self.spillage_amount = [[] for _ in range(self.num_envs)]

        self.record_ee_pose = []
        self.rgb_list = []
        self.depth_list = []

        # create and populate the environments
        for i in range(num_envs):
            # create env
            self.env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(self.env_ptr)

            # add bowl_1
            bowl_pose = gymapi.Transform()
            bowl_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.radians(1.8))
            bowl_pose.p = gymapi.Vec3(0.51, -0.15 , 0.04) 

            self.bowl_1 = self.gym.create_actor(self.env_ptr, self.bowl_asset, bowl_pose, "bowl_1", 0, 0)

            body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.bowl_1)
            # thickness(soft) = 0.0003, thickness(soft) = 0.007
            body_shape_prop[0].thickness = 0.000      # the ratio of the final to initial velocity after the rigid body collides.(but have no idea why it will affect the contact distance) 
            body_shape_prop[0].friction = 0.1
            self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.bowl_1, body_shape_prop)
            
            
            #add bowl_2
            bowl_pose = gymapi.Transform()
            bowl_pose.r = gymapi.Quat(0, 0, 0, 1)
            bowl_pose.p = gymapi.Vec3(0.51, 0.15 , 0.02)   
            self.target_bowl_pose = np.array([bowl_pose.p.x, bowl_pose.p.y, bowl_pose.p.z])
            self.bowl_2 = self.gym.create_actor(self.env_ptr, self.transparent_bowl_asset, bowl_pose, "bowl_2", 0, 0)
            color = gymapi.Vec3(140/255, 227/255, 229/255)
            self.gym.set_rigid_body_color(self.env_ptr, self.bowl_2, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            # body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.bowl)
            # body_shape_prop[0].thickness = 0.005
            # self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.bowl, body_shape_prop)
            
            # add tabel
            table_pose = gymapi.Transform()
            table_pose.r = gymapi.Quat(0, 0, 0, 1)
            table_pose.p = gymapi.Vec3(0.5, -0.5 , -0.2)   
            self.table = self.gym.create_actor(self.env_ptr, self.table_asset, table_pose, "table", 0, 0)
            color = gymapi.Vec3(150/255, 150/255, 140/255)
            self.gym.set_rigid_body_color(self.env_ptr, self.table, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            
            
            # body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.table)
            # body_shape_prop[0].thickness = 0.0005      
            # body_shape_prop[0].friction = 0.5
            # self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.table, body_shape_prop)
            
            #add ball
            if self.grain_type == "solid":
                self.add_solid()
                self.ball_handles[i] = self.ball_handle

            else :
                self.add_soft()

            #add franka
            self.add_franka()

            #add camera_1
            camera_1 = self.gym.create_camera_sensor(self.env_ptr, self.camera_props)
            camera_transform = gymapi.Transform()
            side_cam_pose = np.load('./real_cam_pose/cam2base.npy')

            # 1.004056, -0.047730, 0.766758
            pose = side_cam_pose[:3, 3]

            camera_transform.p = self.cam_pos = gymapi.Vec3(pose[0], pose[1], pose[2])
            #assign
            z = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.radians(180))
            y = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(55.5))
            x = gymapi.Quat.from_axis_angle(gymapi.Vec3(1,0,0), np.radians(-4))

            camera_transform.r = z*y*x
        
            # self.cam_target = gymapi.Vec3(0.5, -0.18, 0.3)
            self.gym.set_camera_transform(camera_1, self.env_ptr, camera_transform)
          
            self.camera_handles[i].append(camera_1)

            # add camera_2(need modify)
            camera_2 = self.gym.create_camera_sensor(self.env_ptr, self.camera_props)
            camera_offset = gymapi.Vec3(0.01, 0.0, 0.11)
            camera_rotation = gymapi.Quat(0.8, 0, 0.8, 0)
          
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
        
    
    def reset_franka(self, init_pose):
        action = init_pose
        action = np.append(action, 0.02)
        action = np.append(action, 0.02)
        franka_init_pose = torch.tensor(action, dtype=torch.float32)
       
        self.dof_state[:, self.franka_dof_index, 0] = franka_init_pose
        self.dof_state[:, self.franka_dof_index, 1] = 0
 

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(self.franka_actor_indices),
            len(self.franka_actor_indices)
        )

    def check_franka(self):
        action = self.joint
        action = np.append(action, 0.02)
        action = np.append(action, 0.02)
        franka_init_pose = torch.tensor(action, dtype=torch.float32)
       
        self.dof_state[:, self.franka_dof_index, 0] = franka_init_pose
        self.dof_state[:, self.franka_dof_index, 1] = 0
 

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(self.franka_actor_indices),
            len(self.franka_actor_indices)
        )

    

    def control_ik(self, dpose, damping=0.05):

        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)

        return u

    def move_generate(self, move_type):

        if move_type == 0:

            begin_rgb = self.rgb_list[0].copy()
      
            images = [begin_rgb] * 5
            depths = [self.depth_list[0].astype(np.float32)] * 5 
            eepose = [self.record_ee_pose[0].astype(np.float32)] * 5
       
            
        else:
            images = self.rgb_list[-5:]  
            depths = self.depth_list[-5 :]
            eepose = self.record_ee_pose[-5:]


        image_array = np.array(images)
        depth_array = np.array(depths) 
        eepose_array = np.array(eepose)
        
        eeposes = self.lfd.run_model(image_array, depth_array, eepose_array)
        return eeposes
    

    def data_collection(self):

        self.collect_time = 1
        action = "rest"
        self.round = [0]*self.num_envs
        self.pos_action = torch.zeros((self.num_envs, 9), dtype=torch.float32)

        self.frame = 0
        dpose_index = 0
        total_action = []

        folder_name = "syl03"
        
        gd_rgb = np.load(f"./dataset/{folder_name}/rgb_front.npy")[0:]
        gd_depth = np.load(f"./dataset/{folder_name}/depth_front.npy")[0:]
        joint_list = np.load(f"./dataset/{folder_name}/joint_states.npy")
        ee_list = np.load(f"./dataset/{folder_name}/ee_pose_euler.npy")

        while not self.gym.query_viewer_has_closed(self.viewer):
     
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.render_all_camera_sensors(self.sim)

            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.frame += 1
            if self.frame <= 30 : 

                if self.frame == 30 :
                    # import time
                    # time.sleep(1)
                    init_joint = joint_list[0]
                    self.reset_franka(init_joint)

            else :   
                
                if self.frame % 1 == 0:
                    self.get_info()
                   
                    
                    if dpose_index % 8 == 0:

                        if len(self.rgb_list) < 5:
                            # self.rgb_list[0] = gd_rgb[0]
                            # self.depth_list[0] = gd_depth[0].astype(np.float32)
                            # plt.imshow(self.depth_list[0])
                            # plt.show()
                            # exit()
                          
                            single_action = self.move_generate(move_type = 0)
                            # single_action = single_action[1:, :]
                            total_action.extend(single_action)
                
        
                        else :
                            # print(f"dpose_index: {dpose_index}")
                            # self.rgb_list = gd_rgb[dpose_index-4 : dpose_index+1].tolist()
                            # self.depth_list = gd_depth[dpose_index-4 : dpose_index+1].astype(np.float32).tolist()
                     
                            single_action = self.move_generate(move_type = 1)
                            # single_action = single_action[1:, :]
                          
                            total_action.extend(single_action)
                          

                    ee_pose = self.gym.get_rigid_transform(self.envs[0], self.ee_handles[0])
                    init_eepose  = self.to_eular(np.array([ee_pose.p.x, ee_pose.p.y, ee_pose.p.z - 0.1, ee_pose.r.w, ee_pose.r.x, ee_pose.r.y, ee_pose.r.z]))

                    dpose = total_action[dpose_index] - init_eepose
                 
                    
                    dpose[0] = 0
                    dpose[1] += 0.035
                    dpose[2] -= 0.03
                    dpose[3] = -dpose[4]
                    dpose[4:] = 0
                    
                    # print(dpose)
             
                 
                    self.pos_action[:, :7] = self.dof_state[:, self.franka_dof_index, 0].squeeze(-1)[:, :7] + self.control_ik(dpose)
                    

                    dpose_index += 1
                    self.dof_state[:, self.franka_dof_index, 0] = self.pos_action.clone()
                    
            
                    # keyboard control

                    # self.keyboard_control()
                    # for evt in self.gym.query_viewer_action_events(self.viewer):
                    #     action = evt.action if (evt.value) > 0 else "rest"
                    # dpose = self.action_space.get(action)

                    '''
                    # gd ee control
                    # print(ee_list[dpose_index-1])
                    # print(ee_list[dpose_index])
                    dpose = ee_list[dpose_index] - ee_list[dpose_index-1]
                    dpose[3] = -dpose[4]
                    dpose[4:6] = 0
   
              
                    self.pos_action[:, :7] = self.dof_state[:, self.franka_dof_index, 0].squeeze(-1)[:, :7] + self.control_ik(dpose*8)
                    dpose_index += 1
                    self.dof_state[:, self.franka_dof_index, 0] = self.pos_action.clone()
                    '''

                    # # gd joint control
                    # joint = joint_list[dpose_index]
                    # action = torch.tensor(np.array(joint), dtype=torch.float32)
                    # self.pos_action[:, :7] =  action
                    # dpose_index += 1
                    # self.dof_state[:, self.franka_dof_index, 0] = self.pos_action.clone()

                    if dpose_index == len(joint_list):

                        folder_path = f"./sim_dataset/{folder_name[:-2]}_{self.radius_ratio}/"
        
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                        
                        joint_path = f'{folder_path}/joint_states.npy'
                        ee_pose_path = f'{folder_path}/ee_pose_qua.npy'
                        rgb_front_path = f'{folder_path}/rgb_front.npy'
                        depth_front_path = f'{folder_path}/depth_front.npy'
                        
                        eepose = self.list_to_nparray(self.record_ee_pose)
                        rgb_front = self.list_to_nparray(self.rgb_list)
                        depth_front = self.list_to_nparray(self.depth_list)

                        '''
                        np.save(joint_path, joint_list)
                        np.save(ee_pose_path, eepose)
                        np.save(rgb_front_path, rgb_front)
                        np.save(depth_front_path, depth_front)

                        with open(f"{folder_path}/setting" , "w") as file:
                            file.write(f"bowl_pose:{self.target_bowl_pose}\n")
                            file.write(f"radius:{self.ball_radius}\n")
                            file.write(f"mass:{self.ball_mass}\n")
                            file.write(f"friction:{self.ball_friction}\n")
                            file.write(f"amount:{self.ball_amount}\n")
                        break
                    '''


            temp = self.dof_state[:, self.franka_dof_index, 0].clone()
            temp[:, -2:] = 0.02

            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(temp),
                gymtorch.unwrap_tensor(self.franka_actor_indices),
                len(self.franka_actor_indices)
            )


            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

            


        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def to_eular(self, pose):

        orientation_list = [pose[3], pose[4], pose[5], pose[6]]
        (roll, pitch, yaw) = transforms3d.euler.quat2euler(pose[3:])
        return np.array([pose[0], pose[1], pose[2],roll,pitch, yaw ])
    
    def get_info(self, env_index = 0):

        # get camera images
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
    
        # get eepose
        ee_pose = self.gym.get_rigid_transform(self.envs[0], self.ee_handles[0])
        ee_pose_arr  = np.array([ee_pose.p.x, ee_pose.p.y, ee_pose.p.z - 0.1, ee_pose.r.w, ee_pose.r.x, ee_pose.r.y, ee_pose.r.z])

        self.record_ee_pose.append(ee_pose_arr)
        
        #get top_image

        top_color_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0][0],  gymapi.IMAGE_COLOR)
        top_color_image = top_color_image.reshape(self.camera_props.height, self.camera_props.width, 4)
    
        self.rgb_list.append(top_color_image[ :, :, :3])
        
        top_depth_image = self.gym.get_camera_image(self.sim, self.envs[env_index], self.camera_handles[env_index][0],  gymapi.IMAGE_DEPTH)
        # self.top_pcd_point_list[env_index].append(np.array(top_pcd.points))
        self.depth_list.append(top_depth_image)



    def compute_camera_intrinsics_matrix(self, image_width = 320, image_heigth = 240, horizontal_fov = 57):
        vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
        horizontal_fov *= np.pi / 180

        f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
        f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)

        K = torch.tensor([[f_x, 0.0, image_width / 2.0], [0.0, f_y, image_heigth / 2.0], [0.0, 0.0, 1.0]], device=self.device, dtype=torch.float)

        return K
    
    def normalize_image(self, img):

        min_val = np.min(img)
        max_val = np.max(img)

        # Normalize the image to 0-1 range
        normalized_img = (img - min_val) / (max_val - min_val)
        normalized_img = 1- normalized_img

        
        return normalized_img

    def list_to_nparray(self, lists):
        temp_array = []


        for i in range(len(lists)):
            temp_array.append(np.array(lists[i]))

        temp = np.stack(temp_array, axis=0)

        return temp

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



if __name__ == "__main__":
    
    issac = IsaacSim()
    issac.data_collection()
