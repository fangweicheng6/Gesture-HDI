import pickle
import time
import numpy as np
import torch
import tqdm
from al3d_det.models import load_data_to_gpu, load_data_to_gpukitti

from pathlib import Path
import copy
from al3d_det.utils.kitti_utils import calibration_kitti
from al3d_det.datasets.kitti import kitti_utils
from al3d_det.datasets.dataset_kitti import DatasetTemplate_KITTI
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录添加到sys.path
sys.path.append(current_dir)
from predict_dir import kitti_dataset


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (
            metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def get_fov_flag(pts_rect, img_shape, calib):
    """
    Args:
        pts_rect:
        img_shape:
        calib:

    Returns:

    """
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag


# 构建输入网络的batch_dict
def build_batch_dict(image_data, pc_data, dataset):
    info = copy.deepcopy(dataset.kitti_infos[0])
    img_shape = info['image']['image_shape']

    calib_path = Path(r"/home/robot/ws_logonet/src/logonet_ros/src/LoGoNet/detection/predict_dir/data/calib/000001.txt")
    assert calib_path.exists()
    calib = calibration_kitti.Calibration(calib_path)

    input_dict = {
        'calib': calib,
    }
    points = pc_data
    if dataset.dataset_cfg.FOV_POINTS_ONLY:
        pts_rect = calib.lidar_to_rect(points[:, 0:3])
        fov_flag = get_fov_flag(pts_rect, img_shape, calib)
        points = points[fov_flag]
    input_dict['points'] = points

    input_dict['images'] = image_data

    input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

    input_dict['calib'] = calib
    input_dict['image_shape'] = img_shape

    data_dict = input_dict
    data_dict = dataset.point_feature_encoder.forward(data_dict)

    data_dict = dataset.data_processor.forward(data_dict)
    data_dict['batch_size'] = 1

    batch_dict = []
    batch_dict.append(data_dict)
    batch_dict = DatasetTemplate_KITTI.collate_batch(batch_dict)
    # print(f"ok:{batch_dict}")
    return batch_dict


# def visualize_and_save(image_data, pc_data, annos, output_image_path, output_pc_path):
#     # 定义颜色列表
#     colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

#     # 创建图像的可视化
#     fig, ax = plt.subplots(1, 1, figsize=(12, 8))
#     ax.imshow(image_data)

#     # 绘制边界框
#     for i, (bbox, name) in enumerate(zip(annos[0]['bbox'], annos[0]['name'])):
#         color = colors[i % len(colors)]
#         rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor=color, facecolor='none')
#         ax.add_patch(rect)
#         ax.text(bbox[0], bbox[1], name, verticalalignment='bottom', horizontalalignment='left', color=color, fontsize=12, weight='bold')

#     # 创建目标目录（如果不存在）
#     os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

#     # 保存图像
#     plt.savefig(output_image_path)
#     # plt.show()
#     plt.close(fig)

#     # 使用matplotlib可视化点云
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(pc_data[:, 0], pc_data[:, 1], pc_data[:, 2], c='b', marker='o')

#     # 绘制边界框并添加类别标签
#     for i, (box, name) in enumerate(zip(annos[0]['boxes_lidar'], annos[0]['name'])):
#         color = colors[i % len(colors)]
#         # 提取位置和尺寸
#         x, y, z, l, w, h, _ = box
#         # 计算边界框的8个顶点
#         x_corners = [x - l / 2, x + l / 2]
#         y_corners = [y - w / 2, y + w / 2]
#         z_corners = [z - h / 2, z + h / 2]

#         # 绘制边界框的线
#         for x_corner in x_corners:
#             for y_corner in y_corners:
#                 ax.plot([x_corner, x_corner], [y_corner, y_corner], z_corners, color=color)
#         for x_corner in x_corners:
#             for z_corner in z_corners:
#                 ax.plot([x_corner, x_corner], y_corners, [z_corner, z_corner], color=color)
#         for y_corner in y_corners:
#             for z_corner in z_corners:
#                 ax.plot(x_corners, [y_corner, y_corner], [z_corner, z_corner], color=color)

#         # 添加类别标签
#         ax.text(x, y, z + h / 2, name, color=color, fontsize=12, weight='bold')

#     # 设置标签
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     # 设置视角
#     ax.view_init(elev=20, azim=240)

#     # 创建目标目录（如果不存在）
#     os.makedirs(os.path.dirname(output_pc_path), exist_ok=True)

#     # 保存点云图像
#     plt.savefig(output_pc_path)
#     # plt.show()  # 显示点云图像
#     plt.close(fig)


# 点云坐标系
# def visualize_and_save(image_data, pc_data, annos, output_image_path, output_pc_path):
#     # 定义颜色列表
#     colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
#
#     # 创建图像的可视化
#     fig, ax = plt.subplots(1, 1, figsize=(12, 8))
#     ax.imshow(image_data)
#
#     # 绘制边界框
#     for i, (bbox, name) in enumerate(zip(annos[0]['bbox'], annos[0]['name'])):
#         color = colors[i % len(colors)]
#         rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor=color, facecolor='none')
#         ax.add_patch(rect)
#         ax.text(bbox[0], bbox[1], name, verticalalignment='bottom', horizontalalignment='left', color=color, fontsize=12, weight='bold')
#
#     # 创建目标目录（如果不存在）
#     os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
#
#     # 保存图像
#     plt.savefig(output_image_path)
#     # plt.show()  # 显示图像
#     plt.close(fig)
#
#     # 使用matplotlib可视化点云
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(pc_data[:, 0], pc_data[:, 1], pc_data[:, 2], c='b', marker='o')
#
#     # 绘制边界框并添加类别标签
#     for i, (box, name) in enumerate(zip(annos[0]['boxes_lidar'], annos[0]['name'])):
#         color = colors[i % len(colors)]
#         # 提取位置和尺寸
#         x, y, z, l, w, h, _ = box
#         # 计算边界框的8个顶点
#         x_corners = [x - l / 2, x + l / 2]
#         y_corners = [y - w / 2, y + w / 2]
#         z_corners = [z - h / 2, z + h / 2]
#
#         # 绘制边界框的线
#         for x_corner in x_corners:
#             for y_corner in y_corners:
#                 ax.plot([x_corner, x_corner], [y_corner, y_corner], z_corners, color=color)
#         for x_corner in x_corners:
#             for z_corner in z_corners:
#                 ax.plot([x_corner, x_corner], y_corners, [z_corner, z_corner], color=color)
#         for y_corner in y_corners:
#             for z_corner in z_corners:
#                 ax.plot(x_corners, [y_corner, y_corner], [z_corner, z_corner], color=color)
#
#         # 添加类别标签
#         ax.text(x, y, z + h / 2, name, color=color, fontsize=12, weight='bold')
#
#     # # 绘制相机坐标系下的真值框
#     # true_value = [2.5608, -0.0204, 0.2050, 0.61, 1.21, 0.91]  # x, y, z, l, w, h
#     # x, y, z, l, w, h = true_value
#     # x_corners = [x - l / 2, x + l / 2]
#     # y_corners = [y - w / 2, y + w / 2]
#     # z_corners = [z - h / 2, z + h / 2]
#     #
#     # # 绘制绿色边界框的线
#     # for x_corner in x_corners:
#     #     for y_corner in y_corners:
#     #         ax.plot([x_corner, x_corner], [y_corner, y_corner], z_corners, color='g')
#     # for x_corner in x_corners:
#     #     for z_corner in z_corners:
#     #         ax.plot([x_corner, x_corner], y_corners, [z_corner, z_corner], color='g')
#     # for y_corner in y_corners:
#     #     for z_corner in z_corners:
#     #         ax.plot(x_corners, [y_corner, y_corner], [z_corner, z_corner], color='g')
#
#     # 设置标签
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#
#     # 创建目标目录（如果不存在）
#     os.makedirs(os.path.dirname(output_pc_path), exist_ok=True)
#
#     # 保存点云图像
#     plt.savefig(output_pc_path)
#     plt.show()  # 显示点云图像
#     plt.close(fig)


# 可视化相机坐标系下的点云和边界框
# def transform_pc_to_camera_coords(pc_data, Tr_velo_to_cam):
#     """
#     将点云数据从激光雷达坐标系转换到相机坐标系
#     :param pc_data: 点云数据，形状为 (N, 3)
#     :param Tr_velo_to_cam: 转换矩阵，形状为 (3, 4)
#     :return: 转换后的点云数据，形状为 (N, 3)
#     """
#
#     # 扩展为4x4矩阵
#     Tr_velo_to_cam = np.vstack((Tr_velo_to_cam, [0, 0, 0, 1]))
#     # 应用转换矩阵
#     pc_data_cam = np.dot(pc_data, Tr_velo_to_cam.T)
#     return pc_data_cam[:, :3]  # 返回前3列
#
# def visualize_and_save(image_data, pc_data, annos, output_image_path, output_pc_path):
#     # 定义颜色列表
#     colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
#
#     # 创建图像的可视化
#     fig, ax = plt.subplots(1, 1, figsize=(12, 8))
#     ax.imshow(image_data)
#
#     # 绘制边界框
#     for i, (bbox, name) in enumerate(zip(annos[0]['bbox'], annos[0]['name'])):
#         color = colors[i % len(colors)]
#         rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor=color,
#                              facecolor='none')
#         ax.add_patch(rect)
#         ax.text(bbox[0], bbox[1], name, verticalalignment='bottom', horizontalalignment='left', color=color,
#                 fontsize=12, weight='bold')
#
#     # 创建目标目录（如果不存在）
#     os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
#
#     # 保存图像
#     plt.savefig(output_image_path)
#     # plt.show()  # 显示图像
#     plt.close(fig)
#
#     # 使用matplotlib可视化点云
#     # 将点云数据转换到相机坐标系
#     Tr_velo_to_cam = np.array([
#         [2.935230000000e-02, -9.991170000000e-01, 3.006030000000e-02, 1.083130000000e-01],
#         [-1.425960000000e-02, -3.048870000000e-02, -9.994330000000e-01, 9.078080000000e-02],
#         [9.994670000000e-01, 2.890700000000e-02, -1.514200000000e-02, 3.842580000000e-01]
#     ])
#     pc_data = transform_pc_to_camera_coords(pc_data, Tr_velo_to_cam)
#
#     # 使用matplotlib可视化点云
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(pc_data[:, 0], pc_data[:, 1], pc_data[:, 2], c='b', marker='o')
#
#     # 绘制边界框并添加类别标签
#     for i, (loc, dims, name) in enumerate(zip(annos[0]['location'], annos[0]['dimensions'], annos[0]['name'])):
#         color = colors[i % len(colors)]
#         # 提取位置和尺寸
#         x, y, z = loc
#         l, w, h = dims
#         # 计算边界框的8个顶点
#         x_corners = [x - l / 2, x + l / 2]
#         y_corners = [y - w / 2, y + w / 2]
#         z_corners = [z - h / 2, z + h / 2]
#
#         # 绘制边界框的线
#         for x_corner in x_corners:
#             for y_corner in y_corners:
#                 ax.plot([x_corner, x_corner], [y_corner, y_corner], z_corners, color=color)
#         for x_corner in x_corners:
#             for z_corner in z_corners:
#                 ax.plot([x_corner, x_corner], y_corners, [z_corner, z_corner], color=color)
#         for y_corner in y_corners:
#             for z_corner in z_corners:
#                 ax.plot(x_corners, [y_corner, y_corner], [z_corner, z_corner], color=color)
#
#         # 添加类别标签
#         ax.text(x, y, z + h / 2, name, color=color, fontsize=12, weight='bold')
#
#     # 绘制相机坐标系下的真值框
#     true_value = [0.21, -0.15, 2.94, 0.61, 1.21, 0.91]  # x, y, z, l, w, h
#     x, y, z, l, w, h = true_value
#     x_corners = [x - l / 2, x + l / 2]
#     y_corners = [y - w / 2, y + w / 2]
#     z_corners = [z - h / 2, z + h / 2]
#
#     # 绘制绿色边界框的线
#     for x_corner in x_corners:
#         for y_corner in y_corners:
#             ax.plot([x_corner, x_corner], [y_corner, y_corner], z_corners, color='g')
#     for x_corner in x_corners:
#         for z_corner in z_corners:
#             ax.plot([x_corner, x_corner], y_corners, [z_corner, z_corner], color='g')
#     for y_corner in y_corners:
#         for z_corner in z_corners:
#             ax.plot(x_corners, [y_corner, y_corner], [z_corner, z_corner], color='g')
#
#     # 设置标签
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#
#     # 创建目标目录（如果不存在）
#     os.makedirs(os.path.dirname(output_pc_path), exist_ok=True)
#
#     # 保存点云图像
#     plt.savefig(output_pc_path)
#     plt.show()  # 显示点云图像
#     plt.close(fig)


def eval_one_epoch(image_data, pc_data, predict_set, cfg, model, epoch_id, logger, save_to_file=False,
                   result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'data'

    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    # metric = {
    #     'gt_num': 0,
    # }
    # for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
    #     metric['recall_roi_%s' % str(cur_thresh)] = 0
    #     metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = predict_set
    class_names = dataset.class_names

    # logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)

    model.eval()

    # if cfg.LOCAL_RANK == 0:
    #     progress_bar = tqdm.tqdm(total=1, leave=True, desc='predict', dynamic_ncols=True)
    # start_time = time.time()

    batch_dict = build_batch_dict(image_data, pc_data, dataset)

    if 'kitti' in cfg.DATA_CONFIG.DATASET.lower():
        load_data_to_gpukitti(batch_dict)
    else:
        load_data_to_gpu(batch_dict)
    with torch.no_grad():

        start_time1 = time.time()
        
        pred_dicts, ret_dict = model(batch_dict)
        
        end_time1 = time.time()
        spend_time1 = end_time1 - start_time1
        print(f"模型推理时间：{spend_time1}秒")

    # disp_dict = {}
    # pred_dicts[0]['pred_boxes'][:, 6] = 0.0 # 方向角设为0
    # statistics_info(cfg, ret_dict, metric, disp_dict)

    annos = kitti_dataset.KittiDataset.generate_prediction_dicts(
        batch_dict, pred_dicts, class_names,
        output_path="/home/robot/ws_logonet/src/logonet_ros/src/LoGoNet/detection/predict_dir/result/logs"
    )

    # if cfg.LOCAL_RANK == 0:
    #     progress_bar.set_postfix(disp_dict)
    #     progress_bar.update()
    # if cfg.LOCAL_RANK == 0:
    #     progress_bar.close()

    # logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    # sec_per_example = (time.time() - start_time) / 1
    # logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    # logger.info('*************** Begin Visual *****************')

    # output_image_path = "/home/robot/ws_logonet/src/logonet_ros/src/LoGoNet/detection/predict_dir/result/image/"
    # output_pc_path = "/home/robot/ws_logonet/src/logonet_ros/src/LoGoNet/detection/predict_dir/result/pc/"
    # visualize_and_save(image_data, pc_data, annos, output_image_path, output_pc_path)
    # logger.info('*************** Finish Visual *****************')
    return image_data, pc_data, annos


if __name__ == '__main__':
    pass
