#!/usr/bin/env python
import sys
sys.path.append('/home/robot/ws_logonet/devel/lib/python3/dist-packages')

import rospy
# from pynput import keyboard
import os
from sensor_msgs.msg import Image, PointCloud2
import numpy as np
from cv_bridge import CvBridge
import time
from LoGoNet.detection.predict_dir.predict_ROS import load_model, predict
from geometry_msgs.msg import PoseStamped, TwistStamped, Point, Pose, Vector3
from std_msgs.msg import Int32MultiArray
# import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from livox_ros_driver.msg._CustomMsg import CustomMsg
from sensor_msgs import point_cloud2
from logonet_ros.msg import DetectionResult # 确保消息类型已经正确定义
import threading
from scipy.spatial.transform import Rotation as R 
import datetime
from visualization_msgs.msg import Marker
from collections import deque  # 引入deque来实现点云数据队列

# 过滤掉ROS传递的未识别参数
sys.argv = [arg for arg in sys.argv if not arg.startswith('__')]

# 过滤掉ROS传递的未识别参数
sys.argv = [arg for arg in sys.argv if not arg.startswith('__')]

class RealTimeVisualizer:
    
    def __init__(self):
        # 开启交互模式
        plt.ion()

        # 初始化图像窗口
        self.fig_img, self.ax_img = plt.subplots(1, 1, figsize=(12, 8))

        # 初始化点云窗口
        self.fig_pc = plt.figure(figsize=(12, 8))
        self.ax_pc = self.fig_pc.add_subplot(111, projection='3d')

        # 定义颜色列表
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    def visualize_and_save(self, image_data, pc_data, annos):
        # 清除上一帧的内容
        self.ax_img.clear()
        self.ax_pc.clear()

        # 显示新图像
        self.ax_img.imshow(image_data)

        # 绘制图像中的边界框
        for i, (bbox, name) in enumerate(zip(annos[0]['bbox'], annos[0]['name'])):
            color = self.colors[i % len(self.colors)]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                     linewidth=2, edgecolor=color, facecolor='none')
            self.ax_img.add_patch(rect)
            self.ax_img.text(bbox[0], bbox[1], name, verticalalignment='bottom',
                            horizontalalignment='left', color=color, fontsize=12, weight='bold')

        # 刷新图像窗口内容
        # self.fig_img.canvas.draw()
        # plt.pause(0.001)  # 短暂停止以更新显示

        # 在函数内部指定保存目录
        # save_dir = r"/home/robot/ws_logonet/src/logonet_ros/src/pc_30"
        # # 确保保存目录存在
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)

        # 获取当前时间并格式化为年月日时分
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H")

        # 创建新的文件夹名称
        save_dir = os.path.join("/home/robot/ws_logonet/src/logonet_ros/src/inference_result", current_time)

        # 确保文件夹存在
        os.makedirs(save_dir, exist_ok=True)


        # 使用当前时间生成唯一的文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())  # 格式化时间戳
        image_filename = os.path.join(save_dir, f"{timestamp}.png")
        pc_filename = os.path.join(save_dir, f"{timestamp}_pc.png")
        
        # 保存图像（包含边界框）
        self.fig_img.savefig(image_filename)

        # 显示点云数据
        self.ax_pc.scatter(pc_data[:, 0], pc_data[:, 1], pc_data[:, 2], c='b', marker='o')

        # 绘制点云中的边界框并添加类别标签
        for i, (box, name) in enumerate(zip(annos[0]['boxes_lidar'], annos[0]['name'])):
            color = self.colors[i % len(self.colors)]
            x, y, z, l, w, h, _ = box
            x_corners = [x - l / 2, x + l / 2]
            y_corners = [y - w / 2, y + w / 2]
            z_corners = [z - h / 2, z + h / 2]

            for x_corner in x_corners:
                for y_corner in y_corners:
                    self.ax_pc.plot([x_corner, x_corner], [y_corner, y_corner], z_corners, color=color)
            for x_corner in x_corners:
                for z_corner in z_corners:
                    self.ax_pc.plot([x_corner, x_corner], y_corners, [z_corner, z_corner], color=color)
            for y_corner in y_corners:
                for z_corner in z_corners:
                    self.ax_pc.plot(x_corners, [y_corner, y_corner], [z_corner, z_corner], color=color)

            self.ax_pc.text(x, y, z + h / 2, name, color=color, fontsize=12, weight='bold')

        # 设置标签和视角
        self.ax_pc.set_xlabel('X')
        self.ax_pc.set_ylabel('Y')
        self.ax_pc.set_zlabel('Z')
        self.ax_pc.view_init(elev=20, azim=180)

        # 刷新点云窗口内容
        # self.fig_pc.canvas.draw()
        # plt.pause(0.001)  # 短暂停止以更新显示
        # 保存点云图（包含边界框）
        self.fig_pc.savefig(pc_filename)


    def close(self):
        # 关闭交互模式并关闭所有窗口
        plt.ioff()
        plt.close(self.fig_img)
        plt.close(self.fig_pc)


# 最终实验使用的代码 含有运动补偿和EKF

class LoGoNetROS:
    def __init__(self):

        # 使用cvbridge可以实现ROS图像消息和opencv图像格式之间互转
        self.bridge = CvBridge()

        # 加载模型和配置文件
        cfg_file = "/home/robot/ws_logonet/src/logonet_ros/src/LoGoNet/detection/tools/cfgs/det_model_cfgs/kitti/LoGoNet-kitti.yaml"
        ckpt_file = "/home/robot/ws_logonet/src/logonet_ros/src/LoGoNet/weight/checkpoint_epoch_80.pth"
        self.model, self.cfg = load_model(cfg_file, ckpt_file)
        
        # 订阅点云和图像话题
        rospy.Subscriber("/livox/lidar", PointCloud2, self.point_cloud_callback)
        # rospy.Subscriber("/livox/lidar", CustomMsg, self.point_cloud_callback)
        rospy.Subscriber("/zed2/zed_node/left/image_rect_color", Image, self.image_callback) 
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.local_pose_callback)
        
        # 订阅线速度话题
        rospy.Subscriber("/mavros/local_position/velocity_body", TwistStamped, self.vel_callback)
        
        # 初始化点云数据队列和图像数据
        self.accumulated_pc_queue = deque(maxlen=30)  # 点云数据队列，最大存储30帧
        self.accumulated_pc_np = np.empty((0, 4))  # 存储累计的点云数据（初始为空数组）
        self.img_data = None
        
        
        
        # 创建两个锁，一个用于点云数据，一个用于图像数据
        self.pc_lock = threading.Lock()  # 点云数据的锁
        self.img_lock = threading.Lock()  # 图像数据的锁
        
        # EKF 状态变量初始化
        self.x_k_k = np.zeros(3)  # 状态估计 (无人机在雷达坐标系下的位置)
        self.P_k_k = np.eye(3)    # 状态协方差矩阵
        
        self.ifdetection = 0
        self.t_v_last = 0.0  # 上一次时间戳
        self.t_v_cur = 0.0   # 当前时间戳
        self.first_msg = True  # 是否是第一条消息
        self.flag_int_xkk = 0
        self.flag_int_xkk_last = 0
        
        # 创建锁
        self.ekf_lock = threading.Lock()
        self.p_f_L = np.zeros(3)
        
        self.cur_rotation_matrix = np.eye(3)  # 3x3 单位矩阵
        self.cur_p_drone_world = None
        
        self.before_predict_p_drone_world = None
        self.before_predict_R_drone_world = None
        
        self.Tr_imu_to_velo = np.array([
            [9.999916000000e-01, 3.808440000000e-03, -1.217160000000e-03, -4.760699600000e-01],
            [-3.738240000000e-03, 9.985733700000e-01, 5.326556000000e-02, 4.214598000000e-02],
            [1.418200000000e-03, -5.326062000000e-02, 9.985792500000e-01, 1.964053000000e-01],
            [0, 0, 0, 1]
        ])
        self.Tr_velo_to_imu = np.linalg.inv(self.Tr_imu_to_velo)
        
        
        
        # 创建保存点云的文件夹
        self.point_cloud_dir = "/home/robot/ws_logonet/src/logonet_ros/src/pc_30"
        os.makedirs(self.point_cloud_dir, exist_ok=True)
        
        
        
        # 初始化可视化工具，用于发布预测结果
        self.visualizer = RealTimeVisualizer()
        
        self.detection_box_publisher = rospy.Publisher('/logonet_detection_box', Point, queue_size=10)
        self.detection_class_publisher = rospy.Publisher('/logonet_detection_class', Int32MultiArray, queue_size=10)
        
        # 定义 Publisher
        self.ekf_predict_pub = rospy.Publisher('/ekf_predict', Point, queue_size=10)

        self.ekf_update_pub = rospy.Publisher('/ekf_update', Point, queue_size=10)

        self.visual_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)

    def vel_callback(self,msg):
        
        velk = msg  # 接收的速度信息
        
        self.t_v_last = self.t_v_cur
        
        # 获取velk消息的时间戳
        self.t_v_cur = velk.header.stamp.to_sec()
        
        if self.first_msg:
            self.first_msg = False
            return
        
        sum_no = 0  # The total points near the feature point
        
        # EKF预测部分
        if self.ifdetection == 1:
            
            # 初始化标志
            self.flag_int_xkk_last = self.flag_int_xkk
            self.flag_int_xkk = 1
            
            # 初始状态
            if self.flag_int_xkk_last == 0 and self.flag_int_xkk == 1:
                self.x_k_k = self.p_f_L  # 初始化卡尔曼滤波器中的变量
            
            if self.flag_int_xkk == 1:
                omega_s = np.array([velk.twist.angular.x, velk.twist.angular.y, velk.twist.angular.z])
                v_s = np.array([velk.twist.linear.x, velk.twist.linear.y, velk.twist.linear.z])
                
                deltat = self.t_v_cur - self.t_v_last  # 采样时间

                omega_s_hat = np.array([[0, -omega_s[2], omega_s[1]],
                                        [omega_s[2], 0, -omega_s[0]],
                                        [-omega_s[1], omega_s[0], 0]])

                cox = 1
                
                # 观测噪声协方差矩阵 越小说明观测值越准 越依赖模型推理的观测值
                Q_variance = np.diag([cox, cox, cox, cox, cox, cox])
                
                
                with self.ekf_lock:
                    p_hat = np.array([[0, -self.x_k_k[2], self.x_k_k[1]],
                                    [self.x_k_k[2], 0, -self.x_k_k[0]],
                                    [-self.x_k_k[1], self.x_k_k[0], 0]])

                    G_T = -deltat * omega_s_hat + np.eye(3)  # f(p) 对 p 求导
                    
                    H_T = np.hstack([p_hat, -np.eye(3)]) * -deltat

                    # 线性
                    # u_k = np.concatenate([omega_s, v_s])
                    # self.x_k_k = G_T @ self.x_k_k + H_T @ u_k
                    
                    # print("Shape of self.x_k_k:", self.x_k_k.shape)
                    # print("Shape of omega_s:", omega_s.shape)
                    # print("self.x_k_k:", self.x_k_k)
                    # print("omega_s:", omega_s)
                    
                    # 非线性预测公式
                    # np.cross叉乘 用于计算由角速度产生的线速度
                    self.x_k_k = deltat * (np.cross(self.x_k_k, omega_s) - v_s) + self.x_k_k
                    self.P_k_k = G_T @ self.P_k_k @ G_T.T + H_T @ Q_variance @ H_T.T

                
                # 计算逆矩阵得到从 LiDAR 到 IMU
                Tr_velo_to_imu = self.Tr_velo_to_imu

                # 将 class_names 转换为 int32 类型标签
                # msg = DetectionResult()
                # msg.class_names = []
                

                lidar_center = np.append(self.x_k_k, 1)
                imu_center = Tr_velo_to_imu @ lidar_center

                # 打印转换后imu_centers
                # print("EKF Predict IMU Centers:", imu_center[:3], "\n")

                # 创建 Point 消息
                box_msg = Point()
                
                box_msg.x = imu_center[0]  # x 坐标
                box_msg.y = imu_center[1]  # y 坐标
                box_msg.z = imu_center[2]  # z 坐标

                # 发布消息
                self.detection_box_publisher.publish(box_msg)

                self.ekf_test_predict(imu_center[:3])


    def ekf_update(self, box_centor_lidar0, class_msg, box_msg):
        
        rospy.loginfo("ekf_update!!")

        try:
            # 检查位姿数据是否有效
            if self.cur_p_drone_world is None or self.before_predict_p_drone_world is None:
                
                # 打印具体的 None 变量
                if self.cur_p_drone_world is None:
                    rospy.logwarn("self.cur_p_drone_world 是 None")
                if self.before_predict_p_drone_world is None:
                    rospy.logwarn("self.before_predict_p_drone_world 是 None")

                rospy.logwarn("无人机位姿数据未初始化，跳过 EKF 更新")
                return

            # 运动补偿 0是模型推理前的box状态 1是当前时刻机体的状态
            # t_body1_to_world =  self.cur_p_drone_world

            # 运动补偿
            t_body1_to_world = self.cur_p_drone_world  # 直接使用 numpy.ndarray
            R_body1_to_world = self.cur_rotation_matrix  # 3x3
                
            t_body0_to_world = self.before_predict_p_drone_world  # 直接使用 numpy.ndarray
            R_body0_to_world = self.before_predict_R_drone_world # 3x3

            
            box_centor_lidar0 = np.append(box_centor_lidar0, 1)
            
            Tr_imu_to_velo = self.Tr_imu_to_velo
            Tr_velo_to_imu = self.Tr_velo_to_imu

            # 1. 将目标从 lidar0 系转换到 imu0 系
            box_centor_imu0 = Tr_velo_to_imu @ box_centor_lidar0
            # print('box_centor_imu0: ', box_centor_imu0)

            # 2. 将目标从 imu0 系转换到世界系
            box_centor_world = R_body0_to_world @ box_centor_imu0[:3] + t_body0_to_world
            box_centor_world = np.append(box_centor_world, 1)  # 转换为齐次坐标 
            # print('box_centor_world: ', box_centor_world)  
            # 3. 将目标从世界系转换到 imu1 系
            R_body1_to_world_inv = np.linalg.inv(R_body1_to_world)
            box_centor_imu1 = R_body1_to_world_inv @ (box_centor_world[:3] - t_body1_to_world)
            box_centor_imu1 = np.append(box_centor_imu1, 1)  # 转换为齐次坐标
            # print('box_centor_imu1: ', box_centor_imu1)
            # 4. 将目标从 imu1 系转换到 lidar1 系     
            box_centor_lidar1 = Tr_imu_to_velo @ box_centor_imu1
            
            rect_uav_pose_lidar = box_centor_lidar1[:3]

            
            # 20250305调试代码
            # rect_uav_pose_lidar = np.append(rect_uav_pose_lidar, 1)
            # imu_center_test = Tr_velo_to_imu @ rect_uav_pose_lidar
            # self.ekf_test_update(imu_center_test[:3])
            # box_msg.x = box_centor_world[0]  # x 坐标
            # box_msg.y = box_centor_world[1]  # y 坐标
            # box_msg.z = box_centor_world[2]
            # self.detection_box_publisher.publish(box_msg) # 改成经过坐标系转换成世界系下的检测位置
               


            # 初始化阶段
            if self.flag_int_xkk == 0:
                rospy.loginfo("initialization!")
                self.p_f_L = rect_uav_pose_lidar      
                self.ifdetection = 1

            # EKF更新部分
            with self.ekf_lock:
                C_T = np.eye(3)  # 观测矩阵
                y = np.zeros(3)  # 观测残差
                z_k = np.zeros(3)  # 观测值
                S = np.zeros((3, 3))  # 残差协方差
                K = np.zeros((3, 3))  # 卡尔曼增益
                
                self.p_f_L = rect_uav_pose_lidar

                z_k = self.p_f_L  # 模型输出
                
                #  过程噪声协方差矩阵 R_variance 表示预测模型的不确定性 越小ekf越信任ekf预测的值
                R_variance = np.array([[0.25, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])
                
                y = z_k - C_T @ self.x_k_k
                S = C_T @ self.P_k_k @ C_T.T + R_variance
                K = self.P_k_k @ C_T.T @ np.linalg.inv(S)
                I_KH = np.eye(3) - K @ C_T 
                
                # 更新
                self.x_k_k = self.x_k_k + K @ y
                self.P_k_k = I_KH @ self.P_k_k @ I_KH.T + K @ R_variance @ K.T
                
                # 计算逆矩阵得到从 LiDAR 到 IMU
                Tr_velo_to_imu = self.Tr_velo_to_imu

                lidar_center = np.append(self.x_k_k, 1)
                imu_center = Tr_velo_to_imu @ lidar_center

                # # 调试代码
                # rect_uav_pose_lidar = np.append(rect_uav_pose_lidar, 1)
                # imu_center_test = Tr_velo_to_imu @ rect_uav_pose_lidar
                # box_msg.x = box_centor_world[0]  # x 坐标
                # box_msg.y = box_centor_world[1]  # y 坐标
                # box_msg.z = box_centor_world[2]
                # # 打印转换后的 class_names 和 imu_centers
                # print("\nEKF Update Class Names:", class_msg.data)
                # print("EKF Update IMU Centers:", imu_center_test[:3])      
                # print("Model Inference lidar box:",rect_uav_pose_lidar,"\n") 
                # # 发布消息
                # self.detection_class_publisher.publish(class_msg)
                # self.detection_box_publisher.publish(box_msg) # 改成经过坐标系转换成世界系下的检测位置
                # self.ekf_test_update(imu_center_test[:3])  # EKF更新的多次坐标系转换世界系下的目标位置
                

                box_msg.x = imu_center[0]  # x 坐标
                box_msg.y = imu_center[1]  # y 坐标
                box_msg.z = imu_center[2]        

                # 打印转换后的 class_names 和 imu_centers
                print("\nEKF Update Class Names:", class_msg.data)
                print("EKF Update IMU Centers:", imu_center[:3])      
                print("Model Inference lidar box:",rect_uav_pose_lidar,"\n")
                
                # 发布消息
                self.detection_class_publisher.publish(class_msg)
                self.detection_box_publisher.publish(box_msg)

                self.ekf_test_update(imu_center[:3]) 

        except Exception as e:
            rospy.logerr(f"Error in EKF update: {str(e)}")

    def ekf_test_predict(self, imu_center):

        t_body_to_world = self.cur_p_drone_world  # 直接使用 numpy.ndarray
        R_body_to_world = self.cur_rotation_matrix
       
        # 2. 将目标从 imu0 系转换到世界系
        box_centor_world = R_body_to_world @ imu_center[:3] + t_body_to_world

        point_msg = Point()
        point_msg.x = box_centor_world[0]  # x 坐标
        point_msg.y = box_centor_world[1]  # y 坐标
        point_msg.z = box_centor_world[2]  # z 坐标
        
        self.ekf_predict_pub.publish(point_msg)
    
    def ekf_test_update(self, imu_center):

        t_body_to_world = self.cur_p_drone_world  # 直接使用 numpy.ndarray
        R_body_to_world = self.cur_rotation_matrix
       
        # 2. 将目标从 imu0 系转换到世界系
        box_centor_world = R_body_to_world @ imu_center[:3] + t_body_to_world

        point_msg = Point()
        point_msg.x = box_centor_world[0]  # x 坐标
        point_msg.y = box_centor_world[1]  # y 坐标
        point_msg.z = box_centor_world[2]  # z 坐标

        self.ekf_update_pub.publish(point_msg)


    
    def point_cloud_callback(self, msg):

        # 提取PointCloud2中的点云数据
        pc = np.array([[p[0], p[1], p[2], p[3]] for p in
                       point_cloud2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)],
                      dtype=np.float32)

        if pc.shape[0] == 0:
            print("跳过空帧点云")
            return  # 跳过空帧

        # 检查点云数据的列数是否为4（x, y, z, intensity）
        if pc.shape[1] != 4:
            print(f"点云数据列数异常,期望4列,实际{pc.shape[1]}列，跳过该帧")
            return

        # 把点云转到世界系
        Tr_velo_to_imu = self.Tr_velo_to_imu
        pc_homogeneous  = np.hstack([pc[:, :3], np.ones((pc.shape[0], 1))]) 
        pc_body = (Tr_velo_to_imu @ pc_homogeneous.T).T

        if self.cur_p_drone_world is None:
            return
        else:
            t = self.cur_p_drone_world
            r = self.cur_rotation_matrix
            pc_w = (r @ pc_body[:, :3].T).T + t

        pc_w = np.hstack([pc_w[:, :3], pc[:, 3].reshape(-1, 1)])

        with self.pc_lock:  # 加锁，确保只有一个线程能修改点云数据
            # 将新的点云数据加入队列
            self.accumulated_pc_queue.append(pc_w)

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img = img.astype(np.float32) / 255.0
        self.img_data = img


    def local_pose_callback(self, msg):   
        
        # 更新当前无人机位姿
        self.cur_p_drone_world = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
         # 将 Quaternion 对象转换为数组 [x, y, z, w]
        cur_q_drone_world = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
        self.cur_rotation_matrix = R.from_quat(cur_q_drone_world).as_matrix()

        # 打印调试信息
        # rospy.loginfo("cur_drone_position: x = %f, y = %f, z = %f", 
        #               self.cur_p_drone_world.x, self.cur_p_drone_world.y, self.cur_p_drone_world.z)
        # rospy.loginfo("cur_drone_orientation: x = %f, y = %f, z = %f, w = %f", 
        #               self.cur_q_drone_world.x, self.cur_q_drone_world.y, self.cur_q_drone_world.z, self.cur_q_drone_world.w)
     
    
    def rviz_visual_livox_frame(self,box):

        x,y,z,l,w,h = box[:6]   
        marker = Marker()
        marker.header.frame_id = "livox_frame"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "box"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position = Point(x,y,z)
        marker.pose.orientation.w = 1.0 # 表示无旋转
        marker.scale = Vector3(l,w,h)
        
        marker.color.r = 1.0 # 红色
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.5 # 透明度

        self.visual_pub.publish(marker)
    
    def publish_detections(self, annos):

        # 定义类名称到 int32 标签的映射
        class_name_to_label = {
            'Up': 1,
            'Down': 2,
            'Left': 3,
            'Right': 4,
            'Hover': 5
        }

        # 创建 Point 消息
        box_msg = Point()
        
        # 创建 Int32MultiArray 消息
        class_msg = Int32MultiArray()
    

        # 确保 annos[0]['name'] 至少有一个元素
        if len(annos[0]['name']) > 0:
            class_msg.data = [class_name_to_label[annos[0]['name'][0]]]
        else:
            # 如果没有 name，初始化为空列表或其他默认值
            class_msg.data = []

        # 确保 annos[0]['boxes_lidar'] 至少有一个 box
        if len(annos[0]['boxes_lidar']) > 0:
            # 取第一个 box 的 xyz 中心点坐标并赋值给 lidar_center
            first_box = annos[0]['boxes_lidar'][0]
            lidar_center = np.array([first_box[0], first_box[1], first_box[2]])

            # 发布rviz中可视化的 box
            self.rviz_visual_livox_frame(first_box)   
            
        else:
            lidar_center = []
        
        if len(lidar_center) > 0:
            self.ekf_update(lidar_center, class_msg, box_msg)

        # else:
            # 将 imu_centers 转换为 float32 并赋值给 bounding_boxes# 只发布中心点坐标
            # msg.bounding_boxes = lidar_center

            # box_msg.x = None  # x 坐标
            # box_msg.y = None  # y 坐标
            # box_msg.z = None  # z 坐标

            # 打印转换后的 class_names 和 imu_centers
            # print("\nEKF Update Class Names:", class_msg.data)
            # print("EKF Update IMU Centers:", lidar_center, "\n")

            # self.detection_class_publisher.publish(class_msg)
            # # 发布消息
            # self.detection_box_publisher.publish(box_msg)
    
    def visualize_and_save_point_cloud(self, pc_data):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制点云数据
        scatter = ax.scatter(pc_data[:, 0], pc_data[:, 1], pc_data[:, 2], c=pc_data[:, 3], cmap='viridis', marker='o')

        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Reflectivity')

        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 设置视角
        ax.view_init(elev=20, azim=240)

        # 创建目标目录（如果不存在）
        os.makedirs(self.point_cloud_dir, exist_ok=True)

        # 保存点云图像
        file_path = os.path.join(self.point_cloud_dir, f"point_cloud_{rospy.Time.now().to_nsec()}.png")
        plt.savefig(file_path)
        rospy.loginfo(f"Point cloud image saved to {file_path}")

        # 显示点云图像
        # plt.show()
        plt.close(fig)

    def run_prediction(self):
        # 如果点云和图像都准备好，开始推理
        if self.accumulated_pc_queue and self.img_data is not None:

            with self.img_lock:  # 加锁，确保在推理时点云和图像数据不被修改
                image_data = self.img_data

            with self.pc_lock:
                # 将队列中的点云数据转换为 numpy 数组
                pc_queue = list(self.accumulated_pc_queue)

            pc_data = np.vstack(pc_queue)
            # 对累计的点云数据进行裁剪（设置包围盒）
            x_min, x_max = 0, 4.8
            y_min, y_max = -2, 2
            z_min, z_max = -2, 2

            pc_data = pc_data[
                (pc_data[:, 0] >= x_min) & (pc_data[:, 0] <= x_max) &
                (pc_data[:, 1] >= y_min) & (pc_data[:, 1] <= y_max) &
                (pc_data[:, 2] >= z_min) & (pc_data[:, 2] <= z_max)
            ]

            pc_point = pc_data[:, :3]

            # 把世界系的pc转到lidar系
            t = self.cur_p_drone_world
            r = self.cur_rotation_matrix

            Tr_imu_to_velo = self.Tr_imu_to_velo
            pc_body = (r.T @ (pc_point - t).T).T
            pc_body_homogeneous = np.hstack([pc_body, np.ones((pc_body.shape[0], 1))])
        
            pc_lidar = (Tr_imu_to_velo @ pc_body_homogeneous.T).T
            
            # 加上反射强度，得到最终的雷达系下的点云数据
            pc = np.hstack([pc_lidar[:, :3], pc_data[:, 3].reshape(-1, 1)])

            # self.visualize_and_save_point_cloud(pc_data)
            self.before_predict_p_drone_world = self.cur_p_drone_world
            self.before_predict_R_drone_world = self.cur_rotation_matrix
            
            start_time1 = time.time()
            
            result_image_data, result_pc_data, annos = predict(self.model, image_data, pc, self.cfg)

            end_time1 = time.time()
            spend_time1 = end_time1 - start_time1
            print(f"执行一次时间：{spend_time1}秒")

            # 实时更新检测框
            # self.visualizer.visualize_and_save(result_image_data, result_pc_data, annos)

            # 发布预测结果
            self.publish_detections(annos)


if __name__ == '__main__':
    rospy.init_node('logonet_ros_node')  # 初始化一个名为logonet_ros_node的ROS节点
    logonet_ros = LoGoNetROS()  # 实例化LoGoNetROS

    rate = rospy.Rate(10)  # 设置循环的频率为10也就是一秒钟执行10次

    while not rospy.is_shutdown():
        logonet_ros.run_prediction()
        rate.sleep()
