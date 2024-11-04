from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *

import os
import time
import yaml
import torch
import pickle
import numpy as np
import open3d as o3d
import pytorch3d.transforms
from tqdm import tqdm
import cv2
import math
import json
torch.pi = math.pi

#for spacemouse control
# run " sudo chmod 0666 /dev/hidraw* " to give lisense
import robosuite as suite
from robosuite.utils.input_utils import input2action
from robosuite.devices import SpaceMouse

from BallGenerator import BallGenerator
from WeighingDomainInfo import WeighingDomainInfo


def pose7d_to_matrix(pose7d: torch.Tensor):
    matrix = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4).repeat(pose7d.shape[0], 1, 1)
    matrix[:, :3, :3] = pytorch3d.transforms.quaternion_to_matrix(pose7d[:, [6, 3, 4, 5]])
    matrix[:, :3, 3] = pose7d[:, :3]

    return matrix

def matrix_to_pose_7d(matrix: torch.Tensor):
    pose_7d = torch.zeros((matrix.shape[0], 7), dtype=torch.float32)
    pose_7d[:, 3:] = pytorch3d.transforms.matrix_to_quaternion(matrix[:, :3, :3])[:, [1, 2, 3, 0]]
    pose_7d[:, :3] = matrix[:, :3, 3]

    return pose_7d

class IsaacSim():
    def __init__(self):
        #tool_type : spoon, knife, stir, fork
        self.tool = "spoon"

        # initialize gym
        self.gym = gymapi.acquire_gym()
        self.domainInfo = WeighingDomainInfo(domain_range=None, flag_list=None)

        # create simulator
        self.env_spacing = 1.5
        self.max_episode_length = 195
        self.asset_root = "urdf"
        self.gravity = -1.0

        self.create_sim()
       
        # create viewer using the default camera properties
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        # create sapcemouse
        self.control = SpaceMouse(pos_sensitivity=1, rot_sensitivity=1)

        # # keyboard event
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "reset")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Z, "quit")

        # Look at the first env
        self.cam_pos = gymapi.Vec3(0.7, 0, 1.2)
        self.cam_target = gymapi.Vec3(0, 0, 0)
        self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, self.cam_target)

        
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_dofs, 2)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        _rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state_tensor = gymtorch.wrap_tensor(_rb_state_tensor).view(-1, 13)

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.j_eef = jacobian[:, self.hand_joint_index, :, :7]
        
        

    def create_sim(self):
        
        # parse arguments
        args = gymutil.parse_arguments(description="Joint control Methods Example")

        args.use_gpu = False
        args.use_gpu_pipeline = False
        self.device = 'cpu'
        #self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_envs = 1

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, self.gravity)

        sim_params.dt = 1.0/20
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
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.env_spacing, int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def create_table(self):

        # create table asset
        table_dims = gymapi.Vec3(0.3, 0.8, self.default_height)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self.table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(0.5, 0, 0)
        self.table_pose.r = gymapi.Quat(0, 0, 0, 1)
        
    def create_bowl(self):

        # Load Bowl asset
        file_name = 'bowl/bowl.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.fix_base_link = True
        asset_options.vhacd_params.resolution = 500000
        self.bowl_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        

    def create_bolt(self):

        # Load bolt asset
        file_name = 'grains/bolt.urdf'
        asset_options = gymapi.AssetOptions()
        self.between_ball_space = 0.1
        asset_options.armature = 0.01
        asset_options.vhacd_params.resolution = 500000
        self.bolt_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        


    def create_ball(self):

        self.domainInfo.set_domain_parameter_all_space(radius = 0.005, mass = 0.03, friction = 0)
        self.ball_radius, self.ball_mass, self.ball_friction = self.domainInfo.get_domain_parameters()
        self.between_ball_space = 0.03
        ballGenerator = BallGenerator()
        file_name = 'BallHLS.urdf'
        ballGenerator.generate(file_name=file_name, ball_radius=self.ball_radius, ball_mass=self.ball_mass)
        self.ball_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, gymapi.AssetOptions())

    def set_ball_property(self, env_ptr, ball_pose):
        
        #ball_handle = self.gym.create_actor(env_ptr, self.ball_asset, ball_pose, "grain", 0, 0)
        ball_handle = self.gym.create_actor(env_ptr, self.ball_asset, ball_pose, "grain", 0, 0)
        body_shape_prop = self.gym.get_actor_rigid_shape_properties(env_ptr, ball_handle)

        body_shape_prop[0].friction = self.ball_friction
        body_shape_prop[0].contact_offset = 0.1   # Distance at which contacts are generated
        body_shape_prop[0].rest_offset = 0      # How far objects should come to rest from the surface of this body 
        body_shape_prop[0].restitution = 0      # when two objects hit or collide, the speed at which they move after the collision
        body_shape_prop[0].thickness = 0        # the ratio of the final to initial velocity after the rigid body collides. 
        
        self.gym.set_actor_rigid_shape_properties(env_ptr, ball_handle, body_shape_prop)
        return ball_handle
    
    def add_ball(self, env_ptr):
        #add balls
        self.ball_amount = 12
        c = np.array([115, 78, 48]) / 255.0
        #c = np.array([1, 1, 0]) 
        color = gymapi.Vec3(c[0], c[1], c[2])

        ball_pose = gymapi.Transform()
        z = self.default_height/2 + 0.2
        ball_pose.r = gymapi.Quat(0, 0, 0, 1)
        ball_spacing = self.between_ball_space
        self.ball_handle_list = []
        
        while self.ball_amount > 0:
            y = -0.025
            ran = min(self.ball_amount, 4)
            for j in range(ran):
                x = 0.5
                for k in range(ran):
                    ball_pose.p = gymapi.Vec3(x, y, z)
                    ball_handle = self.set_ball_property(env_ptr, ball_pose)
                    self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                    self.ball_handle_list.append(ball_handle)
                    #self.gym.set_actor_scale(env_ptr, ball_handle, 1.5)
                    x += ball_spacing*0.15
                y += ball_spacing*0.15
            z += ball_spacing*0.2
            self.ball_amount -= 1

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
        self.franka_dof_props["stiffness"][0:7].fill(10.0)
        self.franka_dof_props["damping"][0:7].fill(4.0)
        self.franka_dof_props["stiffness"][7:9].fill(80.0)
        self.franka_dof_props["damping"][7:9].fill(4.0)
        
        #lock joint
        # locked_joints = []
        # for joint_index in locked_joints:
        #     self.franka_dof_props["effort"][joint_index] = 0

        # set default pose
        self.franka_start_pose = gymapi.Transform()
        self.franka_start_pose.p = gymapi.Vec3(0, 0.0, 0.2)
        self.franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    def add_franka(self, i, env_ptr):
        # create franka and set properties
        self.franka_handle = self.gym.create_actor(env_ptr, self.franka_asset, self.franka_start_pose, "franka", i, 4, 2)
        franka_sim_index = self.gym.get_actor_index(env_ptr, self.franka_handle, gymapi.DOMAIN_SIM)
        self.franka_indices.append(franka_sim_index)

        franka_dof_index = [
            self.gym.find_actor_dof_index(env_ptr, self.franka_handle, dof_name, gymapi.DOMAIN_SIM)
            for dof_name in self.franka_dof_names
        ]
     
        self.franka_dof_indices.extend(franka_dof_index)

        self.franka_hand = self.gym.find_actor_rigid_body_handle(env_ptr, self.franka_handle, "panda_hand")

        self.franka_hand_sim_idx = self.gym.find_actor_rigid_body_index(env_ptr, self.franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
        self.franka_hand_indices.append(self.franka_hand_sim_idx)

        self.gym.set_actor_dof_properties(env_ptr, self.franka_handle, self.franka_dof_props)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, 0.75 * -spacing, 0.0)
        upper = gymapi.Vec3(spacing, 0.75 * spacing, spacing)
        self.default_height = 1.1
        self.create_bowl()
        self.create_ball()
        self.create_table()
        self.create_franka()
        self.create_bolt()
    
        # cache some common handles for later use
        self.camera_handles = []
        self.franka_indices, self.kit_indices, self.urdf_indices = [], [], []
        self.franka_dof_indices = []
        self.franka_hand_indices = []
        self.urdf_link_indices = []
        self.envs = []

        #set camera
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1080
        camera_props.height = 720


        # create and populate the environments
        for i in range(num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)

            #add bowl_1
            bowl_pose = gymapi.Transform()
            bowl_pose.r = gymapi.Quat(1, 0, 0, 1)
            bowl_pose.p = gymapi.Vec3(0.5, 0 , self.default_height/2)  
            self.bowl = self.gym.create_actor(env_ptr, self.bowl_asset, bowl_pose, "bowl", 0, 0)

            #add bowl_2
            bowl_pose = gymapi.Transform()
            bowl_pose.r = gymapi.Quat(1, 0, 0, 1)
            bowl_pose.p = gymapi.Vec3(0.5, 0.2 , self.default_height/2)  
            self.bowl = self.gym.create_actor(env_ptr, self.bowl_asset, bowl_pose, "bowl", 0, 0)


            #add tabel
            self.tabel = self.gym.create_actor(env_ptr, self.table_asset, self.table_pose, "table", 0, 0)

            #add ball
            self.add_ball(env_ptr)

            #add franka
            self.add_franka(i, env_ptr)

            #add spoon
            #self.spoon = self.gym.create_actor(env_ptr, self.spoon_asset, self.spoon_pose, "spoon", 0, 0)

            #add camera_1
            cam_pos = gymapi.Vec3(0.7, 0, 0.9)
            cam_target = gymapi.Vec3(0, 0, 0)
            camera_1 = self.gym.create_camera_sensor(env_ptr, camera_props)
            self.gym.set_camera_location(camera_1, env_ptr, cam_pos, cam_target)
            self.camera_handles.append(camera_1)

            #add camera_2
            camera_2 = self.gym.create_camera_sensor(env_ptr, camera_props)
            camera_offset = gymapi.Vec3(0.1, 0, 0)
            camera_rotation = gymapi.Quat(0.75, 0, 0.7, 0)
          
            self.gym.attach_camera_to_body(camera_2, env_ptr, self.franka_hand, gymapi.Transform(camera_offset, camera_rotation),
                                    gymapi.FOLLOW_TRANSFORM)
            self.camera_handles.append(camera_2)

        self.franka_indices = to_torch(self.franka_indices, dtype=torch.long, device=self.device)
        self.franka_dof_indices = to_torch(self.franka_dof_indices, dtype=torch.long, device=self.device)
        self.franka_hand_indices = to_torch(self.franka_hand_indices, dtype=torch.long, device=self.device)
        self.urdf_indices = to_torch(self.urdf_indices, dtype=torch.long, device=self.device)
        self.urdf_link_indices = to_torch(self.urdf_link_indices, dtype=torch.long, device=self.device)
        self.kit_indices = to_torch(self.kit_indices, dtype=torch.long, device=self.device)

    def reset(self):
        #save property
        self.balls_dis = []
        self.scooped_quantity = []
        self.joint_state = []
        self.franka_init_pose = torch.tensor([0, -0.6, 0, -2, 0, 1.5,  0.8,  0.02,  0.02], dtype=torch.float32, device=self.device)
        
        self.dof_state[:, self.franka_dof_indices, 0] = self.franka_init_pose
        self.dof_state[:, self.franka_dof_indices, 1] = 0
        target_tesnsor = self.dof_state[:, :, 0].contiguous()
        self.pos_action = torch.zeros((self.num_envs, 9), dtype=torch.float32, device=self.device)
        self.pos_action[:, 0:9] = target_tesnsor[:, self.franka_dof_indices[0:9]]

        franka_actor_indices = self.franka_indices.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(franka_actor_indices),
            len(franka_actor_indices)
        )

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(target_tesnsor),
            gymtorch.unwrap_tensor(franka_actor_indices),
            len(franka_actor_indices)
        )

        self.frame = 0

    def quantity(self):
        ball_in_spoon = 0
        for ball_handle in self.ball_handle_list:
            body_states = self.gym.get_actor_rigid_body_states(self.envs[0], ball_handle, gymapi.STATE_ALL)
            z = body_states['pose']['p'][0][2]
            if z > 0.65 and self.frame > 10:
                ball_in_spoon += 1
        return ball_in_spoon

    def control_ik(self, dpose, damping=0.05):
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        print(dpose)
        print(j_eef_T)
        print(torch.inverse(self.j_eef @ j_eef_T + lmbda))
        exit()
        return u


    def data_collection(self):
        self.reset()
        self.control.start_control()
        action = ""
        scoop_round = 0
        pre_time = time.time()

        while not self.gym.query_viewer_has_closed(self.viewer):
            
            #print(self.dof_state[:, self.franka_dof_indices, 0])

            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)

            delta = 3
            #space mouse control
            pose = input2action(device=self.control)
            dpose = pose[:6]
            dpose[3], dpose[4] = dpose[4], dpose[3]
            dpose[3] = 0
            dpose = np.array(dpose)
            dpose = torch.Tensor(dpose.reshape(-1, 1)).to(self.device) * delta


            # get camera images
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            if not os.path.exists(f"collected_data/scoop_round_{scoop_round}/top_view"):
                os.makedirs(f"collected_data/scoop_round_{scoop_round}/top_view")
            if not os.path.exists(f"collected_data/scoop_round_{scoop_round}/hand_view"):
                os.makedirs(f"collected_data/scoop_round_{scoop_round}/hand_view")
            import datetime

            now = datetime.datetime.now()
            if self.frame > 10 :
                self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_COLOR, f"collected_data/scoop_round_{scoop_round}/top_view/{str(self.frame)}.png")
                self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.camera_handles[1], gymapi.IMAGE_COLOR, f"collected_data/scoop_round_{scoop_round}/hand_view/{str(self.frame)}.png")
                #self.gym.write_camera_image_to_file(self.sim, self.envs[i], self.camera_handles[0], gymapi.IMAGE_DEPTH, f"collected_data/${now:%Y.%m.%d}/${now:%H.%M.%S}/top_view/depth_{str(self.frame)}.png")
                #self.gym.write_camera_image_to_file(self.sim, self.envs[i], self.camera_handles[1], gymapi.IMAGE_DEPTH, f"collected_data/${now:%Y.%m.%d}/${now:%H.%M.%S}/hand_view/depth_{str(self.frame)}.png")
                self.scooped_quantity.append(self.quantity())
                self.joint_state.append(list(self.dof_state[:, self.franka_dof_indices, 0]))

                for ball_handle in self.ball_handle_list:
                    body_states = self.gym.get_actor_rigid_body_states(self.envs[0], ball_handle, gymapi.STATE_ALL)
                    ball_dis = body_states['pose']['p'][0]
                    self.balls_dis.append(list(ball_dis))

                       
            
            for evt in self.gym.query_viewer_action_events(self.viewer):
                action = evt.action if (evt.value) > 0 else ""

            if action == "reset" : 
                current_time = time.time()
                if current_time - pre_time > 1:
                    scoop_round += 1
                    pre_time = time.time()
                    self.reset()
            elif action == "quit" : 
                break
    
            
            #print(dpose)
            move = self.control_ik(dpose)
            #redefine rotate and scoop
            if abs(self.control_ik(dpose)[0][6]) > 0.1 :
                move = torch.Tensor([[0,0,dpose[1],0,0,0, dpose[5]]])
            elif abs(dpose[4]) > 0.1 :
                move = torch.Tensor([[0,0,0,0,0, -dpose[4], 0]])
            self.pos_action[:, :7] = self.dof_state[:, self.franka_dof_indices, 0].squeeze(-1)[:, :7] + move

       
            test_dof_state = self.dof_state[:, :, 0].contiguous()
            test_dof_state[:, self.franka_dof_indices] = self.pos_action

            franka_actor_indices = self.franka_indices.to(dtype=torch.int32)
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(test_dof_state),
                gymtorch.unwrap_tensor(franka_actor_indices),
                len(franka_actor_indices)
            )

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            self.gym.sync_frame_time(self.sim)

            self.frame += 1

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


if __name__ == "__main__":
    issac = IsaacSim()
    issac.data_collection()