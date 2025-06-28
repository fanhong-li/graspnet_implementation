#%%
import os
os.environ['DISPLAY'] = ':0'
import sys
import numpy as np
import time
import torch
import pyrealsense2 as rs
from PIL import Image
import MinkowskiEngine as ME
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from models.graspnet import GraspNet, pred_decode
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, get_workspace_mask

#%%
# 配置参数
class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

# 默认配置字典
config_dict = {
    'checkpoint_path': '/home/yinzi/graspness_unofficial/weights/minkuresunet_realsense.tar',  # 需要根据实际情况修改
    'dump_dir': './output',
    'seed_feat_dim': 512,
    'camera': 'realsense',
    'num_point': 15000,
    'batch_size': 1,
    'voxel_size': 0.005,
    'collision_thresh': 0.01,
    'voxel_size_cd': 0.01,
    'infer': True,
    'eval': False
}

cfgs = Config(config_dict)
#%%
# RealSense相机初始化
def init_realsense():
    """初始化RealSense相机"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

def get_realsense_data(pipeline):
    """从RealSense获取深度图、彩色图和相机内参"""
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        return None, None, None
    
    # 对齐深度图和彩色图
    align_to = rs.stream.color
    align = rs.align(align_to)
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    
    # 获取深度图像和彩色图像
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    
    # 获取相机内参
    depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    camera = CameraInfo(1280, 720, 
                       depth_intrinsics.fx, depth_intrinsics.fy,
                       depth_intrinsics.ppx, depth_intrinsics.ppy,
                       1000.0)  # RealSense深度单位为毫米
    
    return depth_image, color_image, camera

def process_realsense_data(depth_image, color_image, camera, num_points=15000, voxel_size=0.005):
    """处理RealSense数据生成模型输入，并添加RGB颜色信息"""
    # 生成点云
    cloud = create_point_cloud_from_depth_image(depth_image, camera, organized=True)
    
    # 获取有效点
    depth_mask = (depth_image > 0)
    cloud_masked = cloud[depth_mask]
    
    # 获取对应的RGB颜色
    colors_masked = color_image[depth_mask]  # 形状为 (N, 3)，BGR格式
    # 转换BGR到RGB
    colors_masked = colors_masked[:, [2, 1, 0]]  # BGR -> RGB
    
    # 简单的工作空间过滤（去除过远和过近的点）
    distance = np.linalg.norm(cloud_masked, axis=1)
    # workspace_mask = (distance > 0.1) & (distance < 1.5)  # 10cm到1.5m范围内
    workspace_mask = (distance > 0.1) & (distance < 1)  # 10cm到1.5m范围内
    cloud_masked = cloud_masked[workspace_mask]
    colors_masked = colors_masked[workspace_mask]
    
    # 随机采样点云
    if len(cloud_masked) > num_points:
        idxs = np.random.choice(len(cloud_masked), num_points, replace=False)
        cloud_sampled = cloud_masked[idxs]
        colors_sampled = colors_masked[idxs]
    else:
        # 如果点数不够，进行重复采样
        if len(cloud_masked) == 0:
            raise ValueError("没有有效的点云数据")
        idxs = np.random.choice(len(cloud_masked), num_points, replace=True)
        cloud_sampled = cloud_masked[idxs]
        colors_sampled = colors_masked[idxs]
    
    # 准备MinkowskiEngine输入
    coords = cloud_sampled / voxel_size
    coords = np.floor(coords).astype(np.int32)
    
    # 创建特征（这里使用简单的全1特征）
    feats = np.ones_like(cloud_sampled).astype(np.float32)
    
    # 进行稀疏量化，获取quantize2original映射
    coords, feats, _, quantize2original = ME.utils.sparse_quantize(
        coords, feats, return_index=True, return_inverse=True)
    
    # 获取量化后对应的点云和颜色
    cloud_quantized = cloud_sampled[quantize2original]
    colors_quantized = colors_sampled[quantize2original]
    
    # 添加batch维度
    batch_coords = np.hstack([np.zeros((len(coords), 1), dtype=np.int32), coords])
    
    # 转换为torch tensor并添加batch维度
    batch_data = {
        'point_clouds': torch.from_numpy(cloud_quantized).float().unsqueeze(0),  # B x N x 3
        'coors': torch.from_numpy(batch_coords).int(),
        'feats': torch.from_numpy(feats).float(),
        'quantize2original': quantize2original  # Already a tensor
    }
    
    return batch_data, cloud_masked, colors_masked

#%%
# 初始化RealSense
print("初始化RealSense相机...")
pipeline = init_realsense()

# 等待相机稳定
print("等待相机稳定...")
for i in range(10):
    frames = pipeline.wait_for_frames()
    time.sleep(0.1)

# 获取实时数据
print("获取RealSense数据...")
depth_image, color_image, camera = get_realsense_data(pipeline)
if depth_image is None:
    raise RuntimeError("无法获取RealSense深度数据")

# 处理数据
batch_data, cloud, colors = process_realsense_data(depth_image, color_image, camera, cfgs.num_point, cfgs.voxel_size)
print(f"点云大小: {len(cloud)}")

#%%
# 加载模型
net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# 加载checkpoint
checkpoint = torch.load(cfgs.checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))
net.eval()

#%%
# 移动batch数据到GPU
for key in batch_data:
    if isinstance(batch_data[key], torch.Tensor):
        batch_data[key] = batch_data[key].to(device)

#%%
# 前向推理
with torch.no_grad():
    end_points = net(batch_data)
    grasp_preds = pred_decode(end_points)

#%%
# 处理预测结果
preds = grasp_preds[0].detach().cpu().numpy()
gg = GraspGroup(preds)

# 碰撞检测
if cfgs.collision_thresh > 0:
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]

#%%
# 可视化
import open3d as o3d

gg = gg.nms().sort_by_score()
gg_pick = gg[0:20]
print(f"检测到 {len(gg_pick)} 个抓取姿态")
if len(gg_pick) > 0:
    print('最佳抓取分数:', gg_pick[0].score)

# 转换为Open3D点云并添加颜色
cloud_o3d = o3d.geometry.PointCloud()
cloud_o3d.points = o3d.utility.Vector3dVector(cloud)
# 添加RGB颜色信息（归一化到0-1范围）
cloud_o3d.colors = o3d.utility.Vector3dVector(colors / 255.0)

# 修复左右翻转问题：翻转Y轴和Z轴
trans_mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])


cloud_o3d.transform(trans_mat)

# 获取抓取器并变换
grippers = gg.to_open3d_geometry_list()
for gripper in grippers:
    gripper.transform(trans_mat)

# 可视化
if len(grippers) > 0:
    print("显示所有抓取姿态...")
    o3d.visualization.draw_geometries([*grippers, cloud_o3d])
    print("显示最佳抓取姿态...")
    o3d.visualization.draw_geometries([grippers[0], cloud_o3d])
else:
    print("未检测到有效抓取姿态，仅显示点云")
    o3d.visualization.draw_geometries([cloud_o3d])

# 关闭RealSense
pipeline.stop()
print("RealSense相机已关闭")

#%%