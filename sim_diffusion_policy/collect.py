import rospy
from sensor_msgs.msg import Image as msg_Image
from cv_bridge import CvBridge
import numpy as np
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from PIL import Image
import os
import roslaunch
import cv2
import time
import dynamic_reconfigure.client
from pynput.keyboard import Listener, KeyCode
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math


class LfD():
    def __init__(self, stage):

        # --------#
        # setting #
        #---------#
        self.control_rate = rospy.Rate(10)
        self.demo_length = 250

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
        self.rgb = rospy.Subscriber("/cam/color/image_raw", msg_Image, self.rgb_callback)
        self.depth = rospy.Subscriber("/cam/aligned_depth_to_color/image_raw", msg_Image, self.depth_callback)

        # ---------------#
        # Initialization #
        #--------------- #  
        self.curr_joint = np.array([0., 0., 0., 0., 0., 0., 0.])
        self.curr_pos = np.array([0.5, 0, 0.5])
        self.curr_ori = np.array([1, 0, 0, 0])

        # origin(480,640,3)
        self.rgb_image = np.zeros((480, 640, 3))
        self.depth_image = np.zeros((480, 640))

        self.init_force_mat = np.zeros((16, 3))
        self.ft_f = np.array([0., 0., 0.])
        self.ft_t = np.array([0., 0., 0.])
        self.step_change = 0.1
        
        self.recorded_traj = None
        self.recorded_ori = None

        self.ros_time = 0
        self.file_name = None
        self.end = False
        self.stage = stage
    
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
    
    def rgb_callback(self, data):
        bridge = CvBridge()
        self.rgb_image = cv2.resize(np.array(bridge.imgmsg_to_cv2(data, data.encoding)), (640, 480))

    def depth_callback(self, data):
        bridge = CvBridge()
        self.depth_image = cv2.resize(np.array(bridge.imgmsg_to_cv2(data, data.encoding)), (640, 480))
    
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
    def homing(self, goal_pose):
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

        rgb = Image.fromarray(np.uint8(self.rgb_image))
        rgb.save('now.png')
    
    # ------ #
    # Record #
    #--------#
    def record_fixlen(self):
        rospy.wait_for_message("/cartesian_pose", PoseStamped)

        self.recorded_traj = self.curr_pos
        self.recorded_ori = self.curr_ori

        print("\n --------- Recording EE Trajectory -----------\n")

        step = 1
        while 1:
            self.recorded_traj = np.c_[self.recorded_traj, self.curr_pos]
            self.recorded_ori  = np.c_[self.recorded_ori, self.curr_ori]
            self.control_rate.sleep()
            print(self.recorded_traj.shape[1])
            if step == self.demo_length:
                break
            step += 1
        rospy.sleep(1)   

        print("\n ------------------ Start to save --------------------\n")
        _ = input("Wait. Press enter.")
        save_name = input("Enter save name: ")
        save_trajectory = np.concatenate((self.recorded_traj, self.recorded_ori), axis=0)
        np.save(save_name, save_trajectory)

        print("Save as {}.".format(save_trajectory))
        print("")
        return save_trajectory
    
    def record(self):
        rospy.wait_for_message("/cartesian_pose", PoseStamped)

        self.recorded_traj = self.curr_pos
        self.recorded_ori = self.curr_ori

        key_pressed = False
        print("\n --------- Recording EE Trajectory, press ""e"" to stop-----------\n")
        self.end = False

        while self.end == False:
            self.recorded_traj = np.c_[self.recorded_traj, self.curr_pos]
            self.recorded_ori  = np.c_[self.recorded_ori, self.curr_ori]
            self.rr.sleep()
            print(self.recorded_traj.shape[1])
        self.end = False
        rospy.sleep(1)   

        # save
        print("\n ------------------ Start to save --------------------\n")
        rospy.sleep(1)
        _ = input("Wait. Press enter.")
        save_name = input("Enter save name: ")
        save_trajectory = np.concatenate((self.recorded_traj, self.recorded_ori), axis=0)
        np.save(save_name, save_trajectory)

        print("Save as {}.npy".format(save_name))
        print("")
        return save_trajectory
    
    # ------ #
    # Replay #
    #--------#
    def replay(self):
        self.set_stiffness(2000, 2000, 2000, 50, 50, 50, 0.0)

        rgb_list = []
        depth_list = []
        ee_pose_qua_list = []
        ee_pose_euler_list = []
        joint_list = []
        ft_list = []

        n_size = self.recorded_traj.shape[-1]
        i = 0
        while (i < n_size and self.end == False):

            print("execute joint i= ", i)
            goal = PoseStamped()
            goal.header.seq = 1
            goal.header.stamp = rospy.Time.now()
            goal.header.frame_id = "map"

            goal.pose.position.x = self.recorded_traj[0][i] 
            goal.pose.position.y = self.recorded_traj[1][i]
            goal.pose.position.z = self.recorded_traj[2][i]

            goal.pose.orientation.w = self.recorded_ori[0][i] 
            goal.pose.orientation.x = self.recorded_ori[1][i] 
            goal.pose.orientation.y = self.recorded_ori[2][i] 
            goal.pose.orientation.z = self.recorded_ori[3][i] 

            self.ee_pub.publish(goal)
            self.rr.sleep()

            # save
            rgb_list.append(self.rgb_image)
            depth_list.append(self.depth_image)

            ee_pose = np.concatenate((self.curr_pos, self.curr_ori), axis=0)
            ee_pose_qua_list.append(ee_pose)

            ee_pose_euler = self.to_eular(ee_pose)
            ee_pose_euler_list.append(ee_pose_euler)

            joint_list.append(self.curr_joint)
            ft = np.concatenate((self.ft_f, self.ft_t), axis = 0)
            ft_list.append(ft)

            i += 1


        assert len(rgb_list)==n_size

        save_name = input("Enter folder name: ")
        if not os.path.exists(save_name):
            os.mkdir(os.path.join('dataset',save_name))
        print("Data len= ", len(rgb_list))

        # save
        np.save('dataset/' + save_name + '/joint_states.npy', np.array(joint_list))   
        np.save('dataset/' + save_name + '/ee_pose_qua.npy', np.array(ee_pose_qua_list))
        np.save('dataset/' + save_name + '/ee_pose_euler.npy', np.array(ee_pose_euler_list))

        np.save('dataset/' + save_name + '/front_rgb.npy', np.array(rgb_list))
        np.save('dataset/' + save_name + '/front_depth.npy', np.array(depth_list))

        np.save('dataset/' + save_name + '/ft.npy', np.array(ft_list))

        print("Save complete.")


# roslaunch realsense2_camera rs_camera.launch camera:=cam_front serial_no:=146322072196 align_depth:=True intial_reset:=True
if __name__ == '__main__':
   
    rospy.init_node('LfD', anonymous=True)
    rospy.loginfo('started')
    rospy.sleep(1)
    print('---------------------------------------------------')
    # record with random time length or fixed time length
    stage = input("Input [record_r/record_f]: ")
    lfd = LfD(stage)

    if stage=='record_f':
        # record
        record_trajectory = lfd.record_fixlen()
        print('Record finish, check the safety bar to blue.')
        # homing
        while 1:
            lfd.homing(record_trajectory[:,0])
            ans = input('Repeat? [y/n]: ')
            if ans=='n':
                break
        # action
        lfd.replay()    
    else:
         # record
        record_trajectory = lfd.record()
        print('Record finish, check the safety bar to blue.')
        # homing
        while 1:
            lfd.homing(record_trajectory[:,0])
            ans = input('Repeat? [y/n]: ')
            if ans=='n':
                break
        # action
        lfd.replay()
            
        
