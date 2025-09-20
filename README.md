# 项目名称：Logonet ROS

## ✨ 项目简介

点云图像多模态手势识别实时控制无人机并实现主动视觉。里面包含真机实验的所有节点文件


## 🚀 快速开始

### 数据集
我用夸克网盘分享了「Dataset-HDI」，点击链接即可保存。打开「夸克APP」，无需下载在线播放视频，畅享原画5倍速，支持电视投屏。
链接：https://pan.quark.cn/s/59ab50f6c2f8
提取码：jTU1

### 模型训练
在自建kitti数据集上训练和测试，训练和推理指令参考文档 "ws_logonet\src\logonet_ros\src\LoGoNet\脚本命令.txt"

### 真机实验
具体可参考文档 "Desktop\mpc.txt"
```bash
# 激光雷达和相机启动节点
source ~/ws_catkin/devel/setup.bash
roslaunch livox_ros_driver livox_lidar_rviz.launch
roslaunch zed_wrapper zed2.launch

# 运行logonet网络推理节点
source ~/ws_logonet/devel/setup.bash
conda activate logo
roslaunch logonet_ros logonet.launch

