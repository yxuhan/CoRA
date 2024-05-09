import sys
sys.path.append(".")
sys.path.append("..")

import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F

import trimesh
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import argparse

from cora.mesh_renderer import MeshRenderer


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str)
parser.add_argument('--mesh_uv_path', type=str)
opt, _ = parser.parse_known_args()


left_eye_mask_path = os.path.join(opt.data_root, "diffuse_leye_mask.png")
right_eye_mask_path = os.path.join(opt.data_root, "diffuse_reye_mask.png")
left_save_dir = os.path.join(opt.data_root, "leye_mask")
right_save_dir = os.path.join(opt.data_root, "reye_mask")
meta_file_path = os.path.join(opt.data_root, "transforms.json")
mesh_uv_path = opt.mesh_uv_path
device = "cuda:0"

os.makedirs(left_save_dir, exist_ok=True)
os.makedirs(right_save_dir, exist_ok=True)

# load dataset
def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ],dtype=pose.dtype)
    return new_pose

# load extrinsics
with open(meta_file_path, 'r') as f:
    meta = json.load(f)
frames = meta["frames"]
cam2world, img_name_list = [], []
num_frames = len(frames)
for f_id in range(num_frames):
    cur_pose = np.array(frames[f_id]['transform_matrix'], dtype=np.float32)
    cur_pose = nerf_matrix_to_ngp(cur_pose, scale=1.)
    cam2world.append(cur_pose)
    img_name = os.path.basename(frames[f_id]['file_path'])
    img_name_list.append(img_name)
cam2world = np.stack(cam2world, axis=0).astype(np.float32)  # [nf,4,4]
cam2world = torch.from_numpy(cam2world).to(device)  # [nf,4,4]
extrin = torch.inverse(cam2world)

# load intrinsics
HEIGHT = int(meta['h'])
WIDTH = int(meta['w'])
intrinsic = np.eye(3, dtype=np.float32)
intrinsic[0, 0] = meta['fl_x']
intrinsic[1, 1] = meta['fl_y']
intrinsic[0, 2] = meta['cx']
intrinsic[1, 2] = meta['cy']
intrinsic = torch.from_numpy(intrinsic).to(device)
intrinsic[0] /= WIDTH
intrinsic[1] /= HEIGHT
intrinsic = intrinsic[None, ...]

# load mesh
mesh = trimesh.load_mesh(mesh_uv_path)
uv = torch.from_numpy(mesh.visual.uv).to(device).float()  # [v,2]
vertices = torch.from_numpy(mesh.vertices).to(device).float()  # [v,3]
faces = torch.from_numpy(mesh.faces).to(device)
mesh_dict = {
    "vertice": vertices[None, ...].repeat(num_frames, 1, 1),
    "faces": faces[None, ...].repeat(num_frames, 1, 1),
    "attributes": uv[None, ...].repeat(num_frames, 1, 1),
    "size": (HEIGHT, WIDTH),
}

# load uv mask
uv_left_eye_mask = transforms.ToTensor()(Image.open(left_eye_mask_path))
uv_right_eye_mask = transforms.ToTensor()(Image.open(right_eye_mask_path))
uv_left_eye_mask = (torch.mean(uv_left_eye_mask, dim=0, keepdim=True) == 1.).float().to(device)  # [1,h,w]
uv_right_eye_mask = (torch.mean(uv_right_eye_mask, dim=0, keepdim=True) == 1.).float().to(device)  # [1,h,w]

# render!
mesh_renderer = MeshRenderer(device)
uv_img, pix_to_face = mesh_renderer.render_mesh(mesh_dict, intrinsic, extrin[:, :-1])

for i in tqdm(range(num_frames)):
    cur_uv_img = uv_img[i:i+1]  # [1,2,h,w]
    grid = cur_uv_img.permute(0, 2, 3, 1)
    grid = 2 * grid - 1
    grid[..., 1] *= -1
    left_eye_mask = F.grid_sample(uv_left_eye_mask[None, ...], grid)
    right_eye_mask = F.grid_sample(uv_right_eye_mask[None, ...], grid)
    save_image(right_eye_mask, os.path.join(right_save_dir, img_name_list[i]))
    save_image(left_eye_mask, os.path.join(left_save_dir, img_name_list[i]))
