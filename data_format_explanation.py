"""
RealSense数据格式与GraspNet输入格式详细说明
"""
import numpy as np
import torch
import MinkowskiEngine as ME
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image

def explain_data_conversion():
    """详细说明RealSense数据到GraspNet输入的转换过程"""
    
    print("=" * 60)
    print("RealSense数据格式与GraspNet输入格式详细说明")
    print("=" * 60)
    
    # 1. RealSense原始数据
    print("\n1. RealSense原始数据格式:")
    print("-" * 30)
    
    # 模拟RealSense深度图
    depth_image = np.random.randint(500, 2000, size=(720, 1280), dtype=np.uint16)
    depth_image[depth_image < 600] = 0  # 模拟无效深度
    
    print(f"深度图形状: {depth_image.shape}")
    print(f"深度图类型: {depth_image.dtype}")
    print(f"深度值范围: {depth_image[depth_image > 0].min()}mm - {depth_image[depth_image > 0].max()}mm")
    print(f"有效像素数: {np.sum(depth_image > 0)} / {depth_image.size}")
    
    # 相机内参
    camera = CameraInfo(1280, 720, 910.0, 910.0, 640.0, 360.0, 1000.0)
    print(f"相机内参: fx={camera.fx}, fy={camera.fy}, cx={camera.cx}, cy={camera.cy}")
    print(f"深度缩放: {camera.scale} (RealSense深度单位为毫米)")
    
    # 2. 点云生成
    print("\n2. 点云生成过程:")
    print("-" * 30)
    
    cloud = create_point_cloud_from_depth_image(depth_image, camera, organized=True)
    print(f"原始点云形状: {cloud.shape}")
    print(f"点云类型: {cloud.dtype}")
    
    # 获取有效点
    depth_mask = (depth_image > 0)
    cloud_masked = cloud[depth_mask]
    print(f"有效点云形状: {cloud_masked.shape}")
    print(f"坐标范围: x[{cloud_masked[:, 0].min():.3f}, {cloud_masked[:, 0].max():.3f}]")
    print(f"          y[{cloud_masked[:, 1].min():.3f}, {cloud_masked[:, 1].max():.3f}]")
    print(f"          z[{cloud_masked[:, 2].min():.3f}, {cloud_masked[:, 2].max():.3f}]")
    
    # 3. 工作空间过滤
    print("\n3. 工作空间过滤:")
    print("-" * 30)
    
    distance = np.linalg.norm(cloud_masked, axis=1)
    workspace_mask = (distance > 0.1) & (distance < 1.5)
    cloud_workspace = cloud_masked[workspace_mask]
    print(f"工作空间过滤后: {cloud_workspace.shape}")
    print(f"距离范围: [{distance[workspace_mask].min():.3f}m, {distance[workspace_mask].max():.3f}m]")
    
    # 4. 随机采样
    print("\n4. 随机采样:")
    print("-" * 30)
    
    num_points = 15000
    if len(cloud_workspace) > num_points:
        idxs = np.random.choice(len(cloud_workspace), num_points, replace=False)
        cloud_sampled = cloud_workspace[idxs]
        print(f"下采样: {len(cloud_workspace)} -> {num_points}")
    else:
        idxs = np.random.choice(len(cloud_workspace), num_points, replace=True)
        cloud_sampled = cloud_workspace[idxs]
        print(f"上采样: {len(cloud_workspace)} -> {num_points}")
    
    print(f"采样后点云形状: {cloud_sampled.shape}")
    
    # 5. MinkowskiEngine量化
    print("\n5. MinkowskiEngine稀疏量化:")
    print("-" * 30)
    
    voxel_size = 0.005  # 5mm体素大小
    coords = cloud_sampled / voxel_size
    coords = np.floor(coords).astype(np.int32)
    print(f"量化前坐标范围: {coords.min(axis=0)} - {coords.max(axis=0)}")
    
    # 创建特征
    feats = np.ones_like(cloud_sampled).astype(np.float32)
    print(f"特征形状: {feats.shape}")
    
    # 稀疏量化
    coords_sparse, feats_sparse, _, quantize2original = ME.utils.sparse_quantize(
        coords, feats, return_index=True, return_inverse=True)
    
    print(f"稀疏量化结果:")
    print(f"  原始点数: {len(coords)}")
    print(f"  稀疏点数: {len(coords_sparse)}")
    print(f"  压缩比: {len(coords_sparse) / len(coords):.3f}")
    print(f"  quantize2original形状: {quantize2original.shape}")
    print(f"  quantize2original类型: {type(quantize2original)}")
    
    # 获取量化后对应的点云
    cloud_quantized = cloud_sampled[quantize2original]
    print(f"量化后点云形状: {cloud_quantized.shape}")
    
    # 6. 最终模型输入格式
    print("\n6. GraspNet模型输入格式:")
    print("-" * 30)
    
    # 添加batch维度
    batch_coords = np.hstack([np.zeros((len(coords_sparse), 1), dtype=np.int32), coords_sparse])
    
    batch_data = {
        'point_clouds': torch.from_numpy(cloud_quantized).float().unsqueeze(0),  # B x N x 3
        'coors': torch.from_numpy(batch_coords).int(),                           # M x 4
        'feats': torch.from_numpy(feats_sparse).float(),                        # M x 3  
        'quantize2original': quantize2original                                   # M
    }
    
    print("模型输入字典keys:", list(batch_data.keys()))
    for key, value in batch_data.items():
        print(f"  {key}:")
        print(f"    形状: {value.shape}")
        print(f"    类型: {value.dtype}")
        if key == 'point_clouds':
            print(f"    含义: 批次×点数×坐标(x,y,z), 单位:米")
        elif key == 'coors':
            print(f"    含义: 稀疏点数×(batch_id,x,y,z), 量化整数坐标")
        elif key == 'feats':
            print(f"    含义: 稀疏点数×特征维度, 这里使用3D坐标作为特征")
        elif key == 'quantize2original':
            print(f"    含义: 稀疏点到原始点的映射索引")
    
    # 7. 数据流向说明
    print("\n7. 模型中的数据流向:")
    print("-" * 30)
    print("1. point_clouds -> 用于最终的抓取位置计算")
    print("2. coors + feats -> MinkowskiEngine稀疏卷积特征提取")
    print("3. 稀疏特征通过quantize2original映射回原始点云")
    print("4. 在原始点云上进行抓取检测和评分")
    
    return batch_data

def compare_formats():
    """对比不同数据格式"""
    print("\n" + "=" * 60)
    print("数据格式对比")
    print("=" * 60)
    
    formats = {
        "RealSense深度图": {
            "形状": "(720, 1280)",
            "类型": "uint16", 
            "单位": "毫米(mm)",
            "范围": "0-65535",
            "用途": "深度传感器原始输出"
        },
        "3D点云": {
            "形状": "(N, 3)",
            "类型": "float32",
            "单位": "米(m)", 
            "范围": "实际物理坐标",
            "用途": "3D空间中的点坐标"
        },
        "量化坐标": {
            "形状": "(M, 3)",
            "类型": "int32",
            "单位": "体素单位",
            "范围": "整数网格坐标", 
            "用途": "稀疏卷积的输入"
        },
        "批次坐标": {
            "形状": "(M, 4)",
            "类型": "int32", 
            "单位": "(batch_id, x, y, z)",
            "范围": "带批次索引的坐标",
            "用途": "MinkowskiEngine输入格式"
        }
    }
    
    for name, info in formats.items():
        print(f"\n{name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    batch_data = explain_data_conversion()
    compare_formats()
    
    print("\n" + "=" * 60)
    print("关键要点总结:")
    print("=" * 60)
    print("1. RealSense输出毫米单位的深度图")
    print("2. 转换为米单位的3D点云")
    print("3. 进行工作空间过滤和采样")
    print("4. 量化为整数坐标进行稀疏卷积")
    print("5. 保持原始点云用于最终抓取计算")
    print("6. 通过quantize2original建立稀疏特征与原始点的对应关系") 