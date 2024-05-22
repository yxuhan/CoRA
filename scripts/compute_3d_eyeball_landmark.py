import os
import torch
import numpy as np
import json
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str, default="data/old_neutral")
opt, _ = parser.parse_known_args()


data_root = opt.data_root
img_root = os.path.join(data_root, "vis_landmarks")
ldm_path = os.path.join(data_root, "2d_eyeball_landmarks.pkl")
meta_file_path = os.path.join(data_root, "transforms.json")
device = "cpu"
total_iter = 1000

ldm_id_list = [
    36, 37, 38, 39, 40, 41,
    42, 43, 44, 45, 46, 47,
]

ldm_info = torch.load(ldm_path)
img_name_list = os.listdir(img_root)

with open(meta_file_path, 'r') as f:
    meta = json.load(f)

HEIGHT = int(meta['h'])
WIDTH = int(meta['w'])

# load intrinsics
intrinsic = np.eye(3, dtype=np.float32)
intrinsic[0, 0] = meta['fl_x']
intrinsic[1, 1] = meta['fl_y']
intrinsic[0, 2] = meta['cx']
intrinsic[1, 2] = meta['cy']
fov = 2 * np.arctan(meta["cx"] / meta["fl_x"]) * 180 / np.pi
intrinsic = torch.from_numpy(intrinsic)

# load cam2world
def nerf_matrix_to_ngp(pose, scale=1.):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ],dtype=pose.dtype)
    return new_pose


def ray_point_dist(ray, origin, point):
    '''
    ray: [n,3]
    origin: [n,3]
    point: [3]
    '''
    origin_point = point - origin  # [n,3]
    origin_point_dist2 = torch.sum(origin_point ** 2, dim=-1)
    origin_proj_dist2 = torch.sum(ray * origin_point, dim=-1) ** 2
    dist2 = origin_point_dist2 - origin_proj_dist2
    return dist2.sum()


def optimize(ray, origin):
    device = ray.device
    point = torch.zeros(3).to(device)
    point.requires_grad_(True)
    optimizer = torch.optim.Adam(params=[{"params": point}], lr=0.1)
    
    for i in tqdm(range(1000)):
        loss = ray_point_dist(ray, origin, point)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return point


frames = meta["frames"]
frames = sorted(frames, key=lambda d: d['file_path'])

ldm_3d_list = []
for ldm_id in ldm_id_list:
    pose_list = []
    ldm_list = []
    score_list = []
    for f_id in range(len(frames)):
        cur_name = os.path.basename(frames[f_id]["file_path"])
        if cur_name not in img_name_list:
            continue
        cur_pose = np.array(frames[f_id]['transform_matrix'], dtype=np.float32)
        cur_pose = nerf_matrix_to_ngp(cur_pose, scale=1.)
        pose_list.append(torch.from_numpy(cur_pose))
        ldm_list.append(ldm_info[cur_name]["landmarks"][ldm_id])
        score_list.append(ldm_info[cur_name]["scores"][ldm_id])

    c2w = torch.stack(pose_list, dim=0)  # [n,4,4]
    ldm_list = torch.tensor(ldm_list).float()  # [n,2]
    score_list = torch.tensor(score_list).float()  # [n]

    # compute rays
    x = ldm_list[:, 0]
    y = ldm_list[:, 1]
    camera_dirs = torch.stack(
        [
            (x - intrinsic[0, 2] + 0.5) / intrinsic[0, 0],
            (y - intrinsic[1, 2] + 0.5) / intrinsic[1, 1],
            torch.ones_like(y),
        ],
        dim=-1,
    )  # [num_rays,3]

    # transform view direction to world space
    directions = torch.matmul(c2w[..., :3, :3], camera_dirs[..., None])[..., 0]
    origins = torch.broadcast_to(c2w[..., :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(
        directions, dim=-1, keepdims=True
    )

    cur_ldm_3d = optimize(viewdirs, origins)
    ldm_3d_list.append(cur_ldm_3d)

ldm_3d_list = torch.stack(ldm_3d_list, dim=0)

print("left eye landmark:")
print(ldm_3d_list[:6])
print("-------------")

print("right eye landmark:")
print(ldm_3d_list[6:])
print("-------------")
