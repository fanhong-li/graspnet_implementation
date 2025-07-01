#!/usr/bin/env python3
import os
import sys
import numpy as np
import time
import torch
import pyrealsense2 as rs
import cv2
import MinkowskiEngine as ME
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval

# 添加路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from models.graspnet import GraspNet, pred_decode
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, get_workspace_mask

# SAM imports
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
    print("SAM is available")
except ImportError:
    SAM_AVAILABLE = False
    print("SAM is not available. Please install segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git")

# 配置参数
SAM_CHECKPOINT = '/home/yinzi/workspace/Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
SAM_MODEL_TYPE = 'vit_h'
GRASPNET_CHECKPOINT = '/home/yinzi/graspness_unofficial/weights/minkuresunet_realsense.tar'

# 抓取检测配置
class GraspConfig:
    def __init__(self):
        self.seed_feat_dim = 512
        self.num_point = 15000
        self.voxel_size = 0.005
        self.collision_thresh = 0.01
        self.voxel_size_cd = 0.01

grasp_cfg = GraspConfig()

class SAMSegmenter:
    def __init__(self, model_type='vit_h', checkpoint_path=None):
        if not SAM_AVAILABLE:
            raise ImportError("SAM is not available. Please install segment-anything.")
        
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.predictor = None
        self.image = None
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_model()
        else:
            print(f"SAM checkpoint not found at {checkpoint_path}")
            print("Please download SAM checkpoint from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
    
    def load_model(self):
        """加载SAM模型"""
        try:
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            sam.to(device)
            self.predictor = SamPredictor(sam)
            print(f"SAM model loaded successfully on {device}")
        except Exception as e:
            print(f"Failed to load SAM model: {e}")
            self.predictor = None
    
    def set_image(self, image):
        """设置要分割的图像"""
        if self.predictor is None:
            return False
        
        # 确保图像是RGB格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            # 如果是BGR，转换为RGB
            if isinstance(image, np.ndarray):
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            print("Image must be RGB format")
            return False
        
        self.image = image_rgb
        self.predictor.set_image(image_rgb)
        return True
    
    def segment_from_points(self, points, labels=None):
        """根据点击点进行分割"""
        if self.predictor is None:
            return None, None, None
        
        if labels is None:
            labels = np.ones(len(points))  # 默认都是前景点
        
        points = np.array(points)
        labels = np.array(labels)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        
        return masks, scores, logits

class OpenCVInteractiveClicker:
    def __init__(self, image, sam_segmenter):
        self.original_image = image.copy()
        self.display_image = image.copy()
        self.sam_segmenter = sam_segmenter
        self.points = []
        self.labels = []
        self.current_mask = None
        self.finished = False
        
        # 设置SAM图像
        success = self.sam_segmenter.set_image(image)
        if not success:
            print("Failed to set image for SAM")
            return
        
        # 窗口名称
        self.window_name = "SAM Interactive Segmentation"
        
        print("OpenCV交互式分割界面:")
        print("- 左键: 点击要分割的物体（前景点）")
        print("- 右键: 点击背景区域（背景点）")
        print("- 按 'r' 键: 重置所有点")
        print("- 按 'q' 或 ESC 键: 完成分割")
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if self.finished:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键：前景点
            self.points.append([x, y])
            self.labels.append(1)
            # 绘制绿色圆圈表示前景点
            cv2.circle(self.display_image, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(self.display_image, (x, y), 10, (255, 255, 255), 2)
            print(f"添加前景点: ({x}, {y})")
            self.update_segmentation()
            
        elif event == cv2.EVENT_RBUTTONDOWN:  # 右键：背景点
            self.points.append([x, y])
            self.labels.append(0)
            # 绘制红色圆圈表示背景点
            cv2.circle(self.display_image, (x, y), 8, (0, 0, 255), -1)
            cv2.circle(self.display_image, (x, y), 10, (255, 255, 255), 2)
            print(f"添加背景点: ({x}, {y})")
            self.update_segmentation()
    
    def update_segmentation(self):
        """更新分割结果"""
        if len(self.points) == 0:
            return
            
        try:
            masks, scores, logits = self.sam_segmenter.segment_from_points(self.points, self.labels)
            
            if masks is not None and len(masks) > 0:
                # 选择得分最高的mask
                best_mask_idx = np.argmax(scores)
                self.current_mask = masks[best_mask_idx]
                
                # 显示分割结果
                self.show_mask_overlay()
                print(f"分割更新完成，得分: {scores[best_mask_idx]:.3f}")
            else:
                print("未能生成有效的分割结果")
        except Exception as e:
            print(f"分割过程中出错: {e}")
    
    def show_mask_overlay(self):
        """显示mask叠加效果"""
        if self.current_mask is None:
            return
        
        # 重新开始显示图像
        self.display_image = self.original_image.copy()
        
        # 创建彩色mask叠加
        mask_overlay = np.zeros_like(self.original_image)
        mask_overlay[self.current_mask] = [0, 255, 0]  # 绿色mask
        
        # 叠加mask到原图像
        alpha = 0.4
        self.display_image = cv2.addWeighted(self.display_image, 1-alpha, mask_overlay, alpha, 0)
        
        # 重新绘制所有点
        for i, (point, label) in enumerate(zip(self.points, self.labels)):
            if label == 1:  # 前景点 - 绿色
                cv2.circle(self.display_image, tuple(point), 8, (0, 255, 0), -1)
                cv2.circle(self.display_image, tuple(point), 10, (255, 255, 255), 2)
            else:  # 背景点 - 红色
                cv2.circle(self.display_image, tuple(point), 8, (0, 0, 255), -1)
                cv2.circle(self.display_image, tuple(point), 10, (255, 255, 255), 2)
    
    def add_instructions_to_image(self):
        """在图像上添加说明文字"""
        instructions = [
            "Left click: Foreground point",
            "Right click: Background point", 
            "Press 'r': Reset points",
            "Press 'q' or ESC: Finish"
        ]
        
        y_offset = 30
        for i, instruction in enumerate(instructions):
            cv2.putText(self.display_image, instruction, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(self.display_image, instruction, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    def reset_points(self):
        """重置所有点"""
        self.points = []
        self.labels = []
        self.current_mask = None
        self.display_image = self.original_image.copy()
        print("重置所有点")
    
    def show_and_wait(self):
        """显示界面并等待用户交互"""
        try:
            # 创建窗口
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            
            while not self.finished:
                # 添加说明文字到图像
                display_with_instructions = self.display_image.copy()
                self.add_instructions_to_image()
                
                # 显示图像
                cv2.imshow(self.window_name, self.display_image)
                
                # 等待按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' 或 ESC
                    self.finished = True
                    print("分割完成")
                elif key == ord('r'):  # 'r' 重置
                    self.reset_points()
            
            # 关闭窗口
            cv2.destroyWindow(self.window_name)
            
            return self.current_mask
            
        except Exception as e:
            print(f"显示界面时出错: {e}")
            cv2.destroyAllWindows()
            return None

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

def process_realsense_data_with_mask(depth_image, color_image, camera, mask=None, num_points=15000, voxel_size=0.005):
    """处理RealSense数据生成模型输入，使用mask过滤特定物体"""
    # 生成点云
    cloud = create_point_cloud_from_depth_image(depth_image, camera, organized=True)
    
    # 获取有效点
    depth_mask = (depth_image > 0)
    
    # 如果提供了分割mask，则进一步过滤
    if mask is not None:
        combined_mask = depth_mask & mask
        print(f"使用SAM分割mask过滤点云")
    else:
        combined_mask = depth_mask
        print("未提供分割mask，使用全部有效深度点")
    
    cloud_masked = cloud[combined_mask]
    colors_masked = color_image[combined_mask]
    
    if len(cloud_masked) == 0:
        raise ValueError("No valid points after masking")
    
    # 转换BGR到RGB
    colors_masked = colors_masked[:, [2, 1, 0]]  # BGR -> RGB
    
    # 简单的工作空间过滤（去除过远和过近的点）
    distance = np.linalg.norm(cloud_masked, axis=1)
    workspace_mask = (distance > 0.1) & (distance < 2)  # 10cm到1m范围内
    cloud_masked = cloud_masked[workspace_mask]
    colors_masked = colors_masked[workspace_mask]
    
    if len(cloud_masked) == 0:
        raise ValueError("No valid points in workspace")
    
    print(f"过滤后的点云大小: {len(cloud_masked)}")
    
    # 随机采样点云
    if len(cloud_masked) > num_points:
        idxs = np.random.choice(len(cloud_masked), num_points, replace=False)
        cloud_sampled = cloud_masked[idxs]
        colors_sampled = colors_masked[idxs]
    else:
        # 如果点数不够，进行重复采样
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
        'quantize2original': quantize2original
    }
    
    return batch_data, cloud_masked, colors_masked

def load_graspnet_model(checkpoint_path, device):
    """加载抓取检测模型"""
    print("加载抓取检测模型...")
    net = GraspNet(seed_feat_dim=grasp_cfg.seed_feat_dim, is_training=False)
    net.to(device)
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"-> 已加载checkpoint {checkpoint_path} (epoch: {start_epoch})")
    net.eval()
    
    return net

def detect_grasps(net, batch_data, cloud, device):
    """检测抓取姿态"""
    print("进行抓取检测...")
    
    # 移动batch数据到GPU
    for key in batch_data:
        if isinstance(batch_data[key], torch.Tensor):
            batch_data[key] = batch_data[key].to(device)
    
    # 前向推理
    with torch.no_grad():
        end_points = net(batch_data)
        grasp_preds = pred_decode(end_points)
    
    # 处理预测结果
    preds = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(preds)
    
    # 碰撞检测
    if grasp_cfg.collision_thresh > 0:
        print("进行碰撞检测...")
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=grasp_cfg.voxel_size_cd)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=grasp_cfg.collision_thresh)
        gg = gg[~collision_mask]
        print(f"碰撞检测后剩余抓取姿态: {len(gg)}")
    
    return gg

def visualize_grasps(cloud, colors, grasps, max_grasps=20):
    """可视化抓取姿态"""
    import open3d as o3d
    
    # 对抓取姿态进行NMS和排序
    gg = grasps.nms().sort_by_score()
    gg_pick = gg[0:max_grasps]
    
    print(f"显示前 {len(gg_pick)} 个最佳抓取姿态")
    if len(gg_pick) > 0:
        print(f'最佳抓取分数: {gg_pick[0].score:.4f}')
    
    # 转换为Open3D点云并添加颜色
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud)
    # 添加RGB颜色信息（归一化到0-1范围）
    cloud_o3d.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    # 修复左右翻转问题：翻转Y轴和Z轴
    trans_mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    cloud_o3d.transform(trans_mat)
    
    # 获取抓取器并变换
    grippers = gg_pick.to_open3d_geometry_list()
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

def save_images(color_image, depth_image, mask=None, output_dir="./output"):
    """保存图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存彩色图像
    cv2.imwrite(os.path.join(output_dir, "color_image.jpg"), color_image)
    print(f"彩色图像已保存到: {output_dir}/color_image.jpg")
    
    # 保存深度图像（转换为可视化格式）
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir, "depth_image.jpg"), depth_colormap)
    print(f"深度图像已保存到: {output_dir}/depth_image.jpg")
    
    # 如果有mask，保存分割结果
    if mask is not None:
        # 保存mask
        mask_image = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, "segmentation_mask.jpg"), mask_image)
        print(f"分割mask已保存到: {output_dir}/segmentation_mask.jpg")
        
        # 保存叠加结果
        overlay_image = color_image.copy()
        mask_overlay = np.zeros_like(color_image)
        mask_overlay[mask] = [0, 255, 0]  # 绿色mask
        overlay_result = cv2.addWeighted(overlay_image, 0.6, mask_overlay, 0.4, 0)
        cv2.imwrite(os.path.join(output_dir, "segmentation_overlay.jpg"), overlay_result)
        print(f"分割叠加结果已保存到: {output_dir}/segmentation_overlay.jpg")

def main():
    # 检查SAM是否可用
    if not SAM_AVAILABLE:
        print("SAM is not available. Please install segment-anything.")
        return
    
    # 设备选择
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化SAM分割器
    print("初始化SAM分割器...")
    sam_segmenter = SAMSegmenter(
        model_type=SAM_MODEL_TYPE,
        checkpoint_path=SAM_CHECKPOINT
    )
    
    if sam_segmenter.predictor is None:
        print("SAM model not loaded. Exiting.")
        return
    
    # 加载抓取检测模型
    try:
        graspnet_model = load_graspnet_model(GRASPNET_CHECKPOINT, device)
    except Exception as e:
        print(f"无法加载抓取检测模型: {e}")
        return
    
    # 初始化RealSense
    print("初始化RealSense相机...")
    try:
        pipeline = init_realsense()
    except Exception as e:
        print(f"无法初始化RealSense相机: {e}")
        return
    
    # 等待相机稳定
    print("等待相机稳定...")
    for i in range(10):
        frames = pipeline.wait_for_frames()
        time.sleep(0.1)
    
    try:
        # 获取实时数据
        print("获取RealSense数据...")
        depth_image, color_image, camera = get_realsense_data(pipeline)
        if depth_image is None or color_image is None:
            raise RuntimeError("无法获取RealSense数据")
        
        print(f"获取到图像 - 彩色图像大小: {color_image.shape}, 深度图像大小: {depth_image.shape}")
        
        # 显示原始图像
        print("显示原始图像...")
        cv2.imshow("Original Color Image", color_image)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("Depth Image", depth_colormap)
        print("按任意键继续进行交互式分割...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 进行交互式分割
        print("开始交互式分割...")
        clicker = OpenCVInteractiveClicker(color_image, sam_segmenter)
        mask = clicker.show_and_wait()
        
        # 保存分割结果
        print("保存分割结果...")
        save_images(color_image, depth_image, mask)
        
        if mask is not None:
            print(f"分割完成! 分割区域包含 {np.sum(mask)} 个像素")
            print(f"分割区域占总像素的 {np.sum(mask) / mask.size * 100:.2f}%")
            
            # 显示分割结果
            overlay_image = color_image.copy()
            mask_overlay = np.zeros_like(color_image)
            mask_overlay[mask] = [0, 255, 0]
            final_result = cv2.addWeighted(overlay_image, 0.6, mask_overlay, 0.4, 0)
            cv2.imshow("Segmentation Result", final_result)
            print("按任意键继续进行抓取检测...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # 使用分割mask进行抓取检测
            print("使用分割结果进行抓取检测...")
            batch_data, cloud, colors = process_realsense_data_with_mask(
                depth_image, color_image, camera, mask, grasp_cfg.num_point, grasp_cfg.voxel_size)
            
            # 检测抓取姿态
            grasps = detect_grasps(graspnet_model, batch_data, cloud, device)
            
            if len(grasps) > 0:
                print(f"检测到 {len(grasps)} 个有效抓取姿态")
                # 可视化抓取姿态
                visualize_grasps(cloud, colors, grasps)
            else:
                print("未检测到有效抓取姿态")
                # 仅显示分割的点云
                import open3d as o3d
                cloud_o3d = o3d.geometry.PointCloud()
                cloud_o3d.points = o3d.utility.Vector3dVector(cloud)
                cloud_o3d.colors = o3d.utility.Vector3dVector(colors / 255.0)
                trans_mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
                cloud_o3d.transform(trans_mat)
                o3d.visualization.draw_geometries([cloud_o3d])
        else:
            print("未获得有效分割结果，跳过抓取检测")
    
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 关闭RealSense
        pipeline.stop()
        print("RealSense相机已关闭")

if __name__ == "__main__":
    main() 