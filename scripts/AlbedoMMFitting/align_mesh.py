'''
rigid transform the coordinate of Metashape camera to ensure the solved mesh in canonical space
'''


import os
import argparse
import numpy as np
import torch
import trimesh
from pytorch3d.transforms import rotation_6d_to_matrix
import json
from scipy.spatial.transform import Rotation

from config import parse_args
from flame.FLAME import FLAME


config = parse_args()
opt = config

data_root = opt.data_root
meta_file_path = os.path.join(data_root, "transforms.json")
meta_file_save_path = os.path.join(data_root, "transforms_aligned.json")

res = torch.load(opt.fitting_pkl_path, map_location="cuda")
for k in res:
    res[k].detach_()

shape = res['shape']
texture = res['texture']
exp = res['exp']
eyes = res['eyes']
eyelids = res['eyelids']
jaw = res['jaw']
t = res['t']
R = res['R']
scale = res['scale']

# compute rotation
rot_mat = rotation_6d_to_matrix(R)
rot_mat = torch.inverse(rot_mat)
rot_mat = rot_mat.cpu().numpy()  # [1,3,3]
rot_mat = rot_mat.astype(np.float64)

# compute translation
flame = FLAME(config).to("cuda")
flame_verts, _, _, _ = flame(
    cameras=torch.eye(3)[None, ...].cuda(),
    shape_params=shape,
    expression_params=exp,
    eye_pose_params=eyes,
    jaw_pose_params=jaw,
    eyelid_params=eyelids,
    rot_params=R,
    trans_params=t,
    scale_params=scale,
)
scale = scale.item()

flame_verts = flame_verts.cpu()[0].numpy()
flame_topo = flame.faces.cpu().numpy()

# trimesh.Trimesh(
#     vertices=flame_verts, faces=flame_topo,
# ).export("flame_unaligned.obj")

flame_verts = (rot_mat @ flame_verts[..., None])[..., 0]

trans = np.mean(flame_verts, axis=0)  # (3,)
flame_verts = flame_verts - trans

# flame_verts = flame_verts / scale * 5.
# trimesh.Trimesh(
#     vertices=flame_verts, faces=flame_topo,
# ).export("flame_aligned.obj")


t1 = -trans[..., None]
r1 = rot_mat[0]

def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ],dtype=pose.dtype)
    return new_pose


with open(meta_file_path, "r") as f:
    info = json.load(f)

for i in range(len(info["frames"])):
    cur_trans_mat = info["frames"][i]["transform_matrix"]
    cam_ext = np.array(cur_trans_mat)
    cam_ext = nerf_matrix_to_ngp(cam_ext, scale=1.)

    # print(cam_ext[:3, 3:])
    # print(Rotation.from_matrix(cam_ext[:3, :3]).as_euler(seq="xyz", degrees=True))
    
    cam_ext = np.linalg.inv(cam_ext)

    rotation = cam_ext[:3, :3]
    trans = cam_ext[:3, 3:]

    r2 = rotation @ r1.T
    t2 = trans - r2 @ t1

    cam_ext_new = np.zeros((4, 4))
    cam_ext_new[:3, :3] = r2
    cam_ext_new[:3, 3:] = t2 / scale * 5.
    cam_ext_new[3, 3] = 1
    new_trans_mat = np.linalg.inv(cam_ext_new)

    info["frames"][i]["transform_matrix"] = new_trans_mat.tolist()

    # print(new_trans_mat[:3, 3:])
    # print(Rotation.from_matrix(new_trans_mat[:3, :3]).as_euler(seq="xyz", degrees=True))
    # print(5 / scale)


# save new transform file
with open(meta_file_save_path, "w") as outfile:
    json.dump(info, outfile, indent=4)
