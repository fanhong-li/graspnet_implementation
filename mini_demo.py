#%%
import os
os.environ['DISPLAY'] = ':0'
import sys
import numpy as np
import argparse
import time
import torch
from torch.utils.data import DataLoader
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import GraspNetDataset, minkowski_collate_fn
from utils.collision_detector import ModelFreeCollisionDetector
#%%
#%%
# 用字典替代argparser
class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

# 默认配置字典
config_dict = {
    'checkpoint_path': '/nvmessd/ssd_share/graspnet/minkuresunet_kinect.tar',  # 需要根据实际情况修改
    'dataset_root': '/nvmessd/ssd_share/graspnet',  # 需要根据实际情况修改
    'dump_dir': './output',
    'seed_feat_dim': 512,
    # 'camera': 'realsense',
    'camera': 'kinect',
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
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass
test_dataset = GraspNetDataset(cfgs.dataset_root, split='test_similar', camera=cfgs.camera, num_points=cfgs.num_point,
                                voxel_size=cfgs.voxel_size, remove_outlier=True, augment=False, load_label=False)
print('Test dataset length: ', len(test_dataset))
scene_list = test_dataset.scene_list()
test_dataloader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False,
                                num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
print('Test dataloader length: ', len(test_dataloader))
#%%
batch_data = next(iter(test_dataloader))
#%%
import matplotlib.pyplot as plt
plt.imshow(batch_data['coors']) 
#%%
net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
# Load checkpoint
checkpoint = torch.load(cfgs.checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))
net.eval()
# %%
# Move batch data to GPU
for key in batch_data:
    if isinstance(batch_data[key], torch.Tensor):
        batch_data[key] = batch_data[key].to(device)
#%%
# Forward pass
with torch.no_grad():
    end_points = net(batch_data)
    grasp_preds = pred_decode(end_points)

# %%
preds = grasp_preds[0].detach().cpu().numpy()
gg = GraspGroup(preds)
# collision detection
if cfgs.collision_thresh > 0:
    cloud = test_dataset.get_data(0, return_raw_cloud=True)
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
# %%
import open3d as o3d

gg = gg.nms().sort_by_score()
gg_pick = gg[0:20]
print(gg_pick.scores)
print('grasp score:', gg_pick[0].score)

# Convert numpy array to Open3D point cloud
cloud_o3d = o3d.geometry.PointCloud()
cloud_o3d.points = o3d.utility.Vector3dVector(cloud)

# Apply transformation
trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
cloud_o3d.transform(trans_mat)

# Get grippers and transform them
grippers = gg.to_open3d_geometry_list()
for gripper in grippers:
    gripper.transform(trans_mat)

# Visualization
o3d.visualization.draw_geometries([*grippers, cloud_o3d])
o3d.visualization.draw_geometries([grippers[0], cloud_o3d])


# %%
