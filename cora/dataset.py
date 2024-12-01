import os
from glob import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import json
import kornia
from torchvision.utils import save_image


def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ],dtype=pose.dtype)
    return new_pose


class MetaShapeDataset(Dataset):
    def __init__(self, cfg, mode="train"):
        super().__init__()
        meta_path = cfg["data"]["meta_path"]
        num_rays = cfg["renderer"]["sample_batch_size"] // cfg["renderer"]["render_n_samples"]
        device = cfg["device"]

        self.id_root = os.path.dirname(meta_path)
        self.num_rays = num_rays
        self.device = device
        self.mode = mode

        transform_name = os.path.basename(meta_path)
        assert transform_name in ["transforms.json", "transforms_aligned.json"]

        meta_file_path = meta_path
        with open(meta_file_path, 'r') as f:
            meta = json.load(f)

        self.HEIGHT = int(meta['h'])
        self.WIDTH = int(meta['w'])

        # load intrinsics
        self.intrinsic = np.eye(3, dtype=np.float32)
        self.intrinsic[0, 0] = meta['fl_x']
        self.intrinsic[1, 1] = meta['fl_y']
        self.intrinsic[0, 2] = meta['cx']
        self.intrinsic[1, 2] = meta['cy']
        self.fov = 2 * np.arctan(meta["cx"] / meta["fl_x"]) * 180 / np.pi
        self.intrinsic = torch.from_numpy(self.intrinsic).to(self.device)
        
        # split dataset
        frames = meta["frames"]
        frames = sorted(frames, key=lambda d: d['file_path'])
        self.cam_idx = list(range(len(frames)))
        if mode in ["val"]:
            frames = frames[::10]
            self.cam_idx = self.cam_idx[::10]
        self.frames = frames
        self.num_frames = len(self.frames)

        image_dir = os.path.join(self.id_root, "image")
        mask_dir = os.path.join(self.id_root, "mask")
        hair_mask_dir = os.path.join(self.id_root, "hair_mask")
        left_eyes_mask_dir = os.path.join(self.id_root, "leye_mask")
        right_eyes_mask_dir = os.path.join(self.id_root, "reye_mask")
        bfm_albedo_dir = os.path.join(self.id_root, "specular_bfm")

        # load per-frame meta information and images
        self.cam2world = []
        self.images, self.masks, self.hair_masks = [], [], []
        self.leye_masks, self.reye_masks, self.bfm_albedos = [], [], []
        for f_id in tqdm(range(self.num_frames)):
            cur_pose = np.array(frames[f_id]['transform_matrix'], dtype=np.float32)
            if transform_name == "transforms.json":
                cur_pose = nerf_matrix_to_ngp(cur_pose, scale=1.)
            self.cam2world.append(cur_pose)
            img_name = os.path.basename(frames[f_id]['file_path'])
            img = self._load_img(os.path.join(image_dir, img_name))
            self.images.append(img)
            mask = self._load_img(os.path.join(mask_dir, img_name))
            self.masks.append(mask[..., :1])
            hair_mask = self._load_img(os.path.join(hair_mask_dir, img_name))
            self.hair_masks.append(hair_mask[..., :1])
            leye_mask = self._load_img(os.path.join(left_eyes_mask_dir, img_name))
            self.leye_masks.append(leye_mask[..., :1])
            reye_mask = self._load_img(os.path.join(right_eyes_mask_dir, img_name))
            self.reye_masks.append(reye_mask[..., :1])
            bfm_albedo = self._load_img(os.path.join(bfm_albedo_dir, img_name))
            self.bfm_albedos.append(bfm_albedo.mean(dim=-1, keepdim=True))

        self.images = torch.stack(self.images, dim=0)
        self.masks = torch.stack(self.masks, dim=0)
        # self.images = self.images * self.masks
        self.hair_masks = torch.stack(self.hair_masks, dim=0)
        self.leye_masks = torch.stack(self.leye_masks, dim=0)
        self.reye_masks = torch.stack(self.reye_masks, dim=0)
        self.bfm_albedos = torch.stack(self.bfm_albedos, dim=0)

        # shrink hair mask by dilating face mask
        ks = cfg["data"].get("dilate_ks", 0)
        if ks > 0:
            face_mask = self.masks - self.hair_masks
            face_mask = face_mask.permute(0, 3, 1, 2)
            kernel = torch.ones(ks, ks).to(self.device)
            hair_masks_shrink = []
            for i in range(len(face_mask)):
                cur_face_mask = face_mask[i: i + 1]
                cur_face_mask = kornia.morphology.dilation(cur_face_mask, kernel)
                hair_mask_before = self.hair_masks[i: i + 1].permute(0, 3, 1, 2)
                hair_mask_after = hair_mask_before * (1 - cur_face_mask)
                hair_masks_shrink.append(hair_mask_after)    
            self.hair_masks = torch.cat(hair_masks_shrink, dim=0).permute(0, 2, 3, 1)
        
        self.cam2world = np.stack(self.cam2world, axis=0).astype(np.float32)  # [nf,4,4]
        self.cam2world = torch.from_numpy(self.cam2world).to(self.device)  # [nf,3,4]

        print("Create [%s] dataset, total [%d] frames" % (self.mode, self.num_frames))

    def __len__(self):
        return self.num_frames
        
    def _load_img(self, pth):
        if os.path.exists(pth):
            img = transforms.ToTensor()(Image.open(pth))
            return img.permute(1, 2, 0).to(self.device)
        else:
            return torch.zeros(self.HEIGHT, self.WIDTH, 3).to(self.device)
    
    def __getitem__(self, index):
        num_rays = self.num_rays
        info = {}

        if self.mode == "train":
            x = torch.randint(0, self.WIDTH, size=(num_rays,)).to(self.device)
            y = torch.randint(0, self.HEIGHT, size=(num_rays,)).to(self.device)
            cam_idx = torch.randint(0, self.num_frames, size=(num_rays,)).to(self.device)
            cam_offset_idx = torch.clone(cam_idx)
        else:
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH),
                torch.arange(self.HEIGHT),
                indexing="xy",
            )
            x = x.flatten().to(self.device)
            y = y.flatten().to(self.device)
            cam_idx = torch.tensor([index]).long().to(self.device)
            cam_offset_idx = torch.tensor([self.cam_idx[index]]).long().to(self.device)
        
        img = self.images[cam_idx, y, x]
        render_mask = self.masks[cam_idx, y, x]
        render_smooth_mask = self.hair_masks[cam_idx, y, x]
        render_diffuse_mask = self.hair_masks[cam_idx, y, x]
        render_left_eye_mask = self.leye_masks[cam_idx, y, x]
        render_right_eye_mask = self.reye_masks[cam_idx, y, x]
        render_bfm_albedo = self.bfm_albedos[cam_idx, y, x]
        
        attr = {
            "pixels": img,
            "mask": render_mask,
            "diff_mask": render_diffuse_mask,
            "smooth_mask": render_smooth_mask,
            "leye_mask": render_left_eye_mask,
            "reye_mask": render_right_eye_mask,
            "bfm_albedo": render_bfm_albedo,
            "c2w": self.cam2world[cam_idx],  # [nray,4,4]
            "cam_idx": cam_idx,
            "cam_offset_idx": cam_offset_idx,
        }

        color_bkgd = torch.tensor([0, 0, 0]).float().to(self.device)
        info["rand_x"] = x
        info["rand_y"] = y
        info["color_bkgd"] = color_bkgd
        info["rand_attr"] = attr
        info["intrinsic"] = self.intrinsic

        return info

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays
    
    def set_mesh_render_mode(self):
        assert self.mode == "train"
        self.mode = "train_mesh"
