'''
this code is heavily borrowed from https://github.com/Zielon/metrical-tracker
'''


import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
from torchvision import transforms
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.io import load_obj
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
import face_alignment
import trimesh
from tqdm import tqdm

from flame.FLAME import FLAME, AlbedoMMFlameTex
from config import parse_args
from mesh_renderer import MeshRenderer
from face_detector import FaceDetector
import util
from render_utils import _apply_shading_burley, _compute_rays


mediapipe_idx = np.load(
    'flame/mediapipe/mediapipe_landmark_embedding.npz', allow_pickle=True, encoding='latin1'
)['landmark_indices'].astype(int)
left_iris_flame = [4597, 4542, 4510, 4603, 4570]
right_iris_flame = [4051, 3996, 3964, 3932, 4028]
left_iris_mp = [468, 469, 470, 471, 472]
right_iris_mp = [473, 474, 475, 476, 477]
I = torch.eye(3)[None].detach()


def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3]],
        [0, 0, 0, 1],
    ],dtype=pose.dtype)
    return new_pose


class FlameFitter:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.data_root = config["data_root"]
        self.save_root = config["save_root"]
        self.light_color = torch.tensor(
            config["light_color"]
        ).to(device)[..., None, None] * config["light_intensity"]
        self.gamma = config["gamma"]

        self.flame = FLAME(config).to(device)
        self.flametex = AlbedoMMFlameTex(config).to(device)
        self.mesh_renderer = MeshRenderer(device)
        self.face_detector_mediapipe = FaceDetector('google')
        # self.face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)
        self.face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)
        
        # load uv and skin texture mask
        mesh_file = 'data/head_template_mesh.obj'
        verts, faces, aux = load_obj(mesh_file)
        uvcoords = aux.verts_uvs  # [v,2]
        uvfaces = faces.textures_idx  # [f,3]
        self.face_uvcoords = uvcoords[uvfaces.reshape(-1)].reshape(-1, 3, 2).to(device)  # [f,3,2]
        self.uv_face_mask = (
            transforms.ToTensor()(Image.open("data/uv_mask_eyes.png"))[None, :3].mean(dim=1, keepdim=True) == 1.
        ).float().to(device)  # [1,3,uh,uw]
        self.uv_eye_mask = (
            transforms.ToTensor()(Image.open("data/hyx_uv_mask_eyes.png"))[None, :3].mean(dim=1, keepdim=True) == 1.
        ).float().to(device)  # [1,3,uh,uw]
        
        self._load_dataset()
        os.makedirs(self.save_root, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.save_root)

    def _load_dataset(self):
        self.images, self.extrinsics, self.landmarks, self.landmarks_dense = [], [], [], []
        self.height, self.width = 960, 720

        img_root = os.path.join(self.data_root, "image")
        meta_file_path = os.path.join(self.data_root, "transforms.json")
        with open(meta_file_path, 'r') as f:
            meta = json.load(f)
        
        # load intrinsic
        intrinsic = np.eye(3, dtype=np.float32)
        intrinsic[0, 0] = meta['fl_x']
        intrinsic[1, 1] = meta['fl_y']
        intrinsic[0, 2] = meta['cx']
        intrinsic[1, 2] = meta['cy']
        intrinsic = torch.from_numpy(intrinsic).to(self.device)
        self.intrinsic = intrinsic
        
        # load extrinsics, images, and landmarks
        frames = meta["frames"]
        steps = len(frames) // 10  # select ~15 views to optimize
        frames = frames[::steps]
        num_frames = len(frames)
        for f_id in tqdm(range(num_frames)):
            img_name = os.path.basename(frames[f_id]['file_path'])
            cur_pose = np.array(frames[f_id]['transform_matrix'], dtype=np.float32)
            cur_pose = nerf_matrix_to_ngp(cur_pose, scale=1.)
            extrin = torch.inverse(torch.from_numpy(cur_pose))

            cur_img_pth = os.path.join(img_root, img_name)
            cur_img = cv2.imread(cur_img_pth)[..., [2, 1, 0]]  # [h,w,3] rgb format
            lmk, dense_lmk = self._process_face(cur_img)
            if lmk is None or dense_lmk is None:  # no face detected
                continue

            self.landmarks.append(torch.from_numpy(lmk).float())
            self.landmarks_dense.append(torch.from_numpy(dense_lmk).float())
            self.extrinsics.append(extrin)
            self.images.append(torch.from_numpy(cur_img).float().contiguous())

        self.images = torch.stack(self.images, dim=0).to(self.device) / 255.  # [b,h,w,3]
        self.images = self.images.permute(0, 3, 1, 2)  # [b,3,h,w]
        self.extrinsics = torch.stack(self.extrinsics, dim=0).to(self.device)  # [b,4,4]
        self.landmarks = torch.stack(self.landmarks, dim=0).to(self.device)  # [b,68,2]
        self.landmarks_dense = torch.stack(self.landmarks_dense, dim=0).to(self.device)  # [b,478,2]
        self.mica_shape = torch.from_numpy(
            np.load(os.path.join(self.data_root, "identity.npy"))
        ).float()[None, ...].to(self.device)
        self.cam2world = torch.inverse(self.extrinsics)  # [b,4,4]
        
        # load camera origins and view-direction images for shading
        self.origins, self.viewdirs = _compute_rays(
            self.cam2world, self.intrinsic[None, ...], self.height, self.width, self.device
        )
        self.viewdirs = self.viewdirs.permute(0, 3, 1, 2)  # [b,3,h,w]
        
        # set mesh
        self.faces = torch.clone(self.flame.faces)[None, ...].repeat(len(self.cam2world), 1, 1)  # [b,f,3]
        self.mesh_dict = {
            "faces": self.faces,  # [b,f,3]
            "face_attributes": self.face_uvcoords[None, ...].repeat(len(self.cam2world), 1, 1, 1),  # [b,f,3,2]
            "size": (self.height, self.width),
        }

    def _process_face(self, image):
        lmks, scores, detected_faces = self.face_detector.get_landmarks_from_image(image, return_landmark_score=True, return_bboxes=True)
        if detected_faces is None:
            lmks = None
        else:
            lmks = lmks[0]
        dense_lmks = self.face_detector_mediapipe.dense(image)
        return lmks, dense_lmks

    def _parse_landmarks(self):
        batch = {}
        images = self.images
        landmarks = self.landmarks
        landmarks_dense = self.landmarks_dense

        lmk_dense_mask = ~(landmarks_dense.sum(2, keepdim=True) == 0)
        lmk_mask = ~(landmarks.sum(2, keepdim=True) == 0)

        left_iris = landmarks_dense[:, left_iris_mp, :]
        right_iris = landmarks_dense[:, right_iris_mp, :]
        mask_left_iris = lmk_dense_mask[:, left_iris_mp, :]
        mask_right_iris = lmk_dense_mask[:, right_iris_mp, :]

        batch['left_iris'] = left_iris
        batch['right_iris'] = right_iris
        batch['mask_left_iris'] = mask_left_iris
        batch['mask_right_iris'] = mask_right_iris

        return images, landmarks, landmarks_dense[:, mediapipe_idx, :2], lmk_dense_mask[:, mediapipe_idx, :], lmk_mask, batch

    @staticmethod
    def reduce_loss(losses):
        all_loss = 0.
        for key in losses.keys():
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        return all_loss

    def transform_points_screen(self, vertices):
        '''
        vertices: [b,v,3]
        '''
        # [b,1,3,3] @ [b,v,3,1] + [b,1,3,1] = [b,v,3,1]
        vertices_cam = self.extrinsics[:, None, :3, :3] @ vertices[..., None] + self.extrinsics[:, None, :3, 3:]
        # [3,3] @ [b,v,3,1] = [b,v,3,1]
        vertices_proj = (self.intrinsic @ vertices_cam)[..., 0]  # [b,v,3]
        vertices_proj = vertices_proj[..., :-1] / vertices_proj[..., -1:]  # [b,62,2]
        return vertices_proj

    def render_img(self, vertices, texture):
        '''
        vertices: [b,v,3]
        '''
        diff_albedo, spec_albedo = self.flametex(texture.repeat(len(vertices), 1))  # [b,3,h,w]
        dist2 = torch.sum((vertices - self.origins[:, None, :]) ** 2, dim=-1, keepdim=True)  # [b,v,1]
        normal = util.vertex_normals(vertices, self.faces)  # [b,v,3]
        attr = torch.cat([normal, dist2], dim=-1)
        self.mesh_dict.update({
            "vertice": vertices,
            "attributes": attr,
        })
        cam_int = torch.clone(self.intrinsic)
        cam_int[0] /= self.width
        cam_int[1] /= self.height
        attr_img, uv, pix_to_face = self.mesh_renderer.render_mesh(
            self.mesh_dict, cam_int[None, ...], self.extrinsics[:, :3]
        )
        uv = uv.permute(0, 2, 3, 1)  # [1,h,w,2]
        uv = 2 * uv - 1
        uv[..., 1] *= -1

        mask = (pix_to_face > -1).float().permute(0, 3, 1, 2)  # [b,1,h,w]
        normal_img = attr_img[:, :3]
        dist2_img = attr_img[:, 3:4] + 1e-6
        frontal_mask = torch.sum(-normal_img * self.viewdirs, dim=1, keepdim=True).float()
        mask = mask * (frontal_mask > 0).float()
        diff_albedo_img = F.grid_sample(diff_albedo, uv)
        spec_albedo_img = F.grid_sample(spec_albedo, uv)
        face_mask = F.grid_sample(self.uv_face_mask.repeat(len(vertices), 1, 1, 1), uv)

        # shading
        spec_albedo_vec = spec_albedo_img.permute(0, 2, 3, 1).reshape(-1, 3).mean(dim=-1, keepdim=True)
        diff_albedo_vec = diff_albedo_img.permute(0, 2, 3, 1).reshape(-1, 3)
        roughness_vec = torch.ones_like(spec_albedo_vec) * 0.5
        viewdir_vec = self.viewdirs.permute(0, 2, 3, 1).reshape(-1, 3)
        normal_vec = normal_img.permute(0, 2, 3, 1).reshape(-1, 3)
        diff_shading, spec_shading = _apply_shading_burley(
            normals=normal_vec,
            view_dirs=viewdir_vec,  # from camera to surface
            light_dirs=viewdir_vec,  # from light to surface
            specular=spec_albedo_vec,
            base_color=diff_albedo_vec,
            roughness=roughness_vec,
        )
        diff_shading_img = diff_shading.reshape(-1, self.height, self.width, 3).permute(0, 3, 1, 2)
        spec_shading_img = spec_shading.reshape(-1, self.height, self.width, 1).permute(0, 3, 1, 2)
        render_img = (diff_shading_img + spec_shading_img) * self.light_color / dist2_img * mask
        return render_img * face_mask

    def optimize_landmarks(self, iters=1000, log_freq=250):
        # create params
        bz = 1
        scale = 6. * torch.ones(1).to(self.device)
        R = matrix_to_rotation_6d(I).to(self.device)
        t = torch.zeros(bz, 3).float().to(self.device)
        shape = self.mica_shape
        exp = torch.zeros(bz, self.config.num_exp_params).float().to(self.device)
        texture = torch.zeros(bz, self.config.tex_params).float().to(self.device)
        eyes = torch.cat([matrix_to_rotation_6d(I), matrix_to_rotation_6d(I)], dim=1).to(self.device)
        jaw = matrix_to_rotation_6d(I).to(self.device)
        eyelids = torch.zeros(bz, 2).float().to(self.device)
        
        # set require grad
        scale.requires_grad_(True)
        R.requires_grad_(True)
        t.requires_grad_(True)
        exp.requires_grad_(True)
        texture.requires_grad_(True)
        eyes.requires_grad_(True)
        jaw.requires_grad_(True)
        eyelids.requires_grad_(True)

        # set optimizers and schedulers
        params = [
            {'params': texture, 'lr': 0.025},
            {'params': exp, 'lr': 0.025},
            {'params': eyes, 'lr': 0.001},
            {'params': eyelids, 'lr': 0.01},
            {'params': R, 'lr': 0.05},
            {'params': t, 'lr': 0.05},
            {'params': scale, 'lr': 0.05},
            {'params': jaw, 'lr': 0.001}
        ]
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=iters//4, gamma=0.3)
        
        for p in tqdm(range(iters)):            
            bs = len(self.extrinsics)
            I6D = matrix_to_rotation_6d(I).to(self.device)
            images, landmarks, landmarks_dense, lmk_dense_mask, lmk_mask, batch = self._parse_landmarks()
            image_lmks68 = landmarks
            image_lmksMP = landmarks_dense
            left_iris = batch['left_iris']
            right_iris = batch['right_iris']
            mask_left_iris = batch['mask_left_iris']
            mask_right_iris = batch['mask_right_iris']
            vertices, lmk68, lmkMP, act_trans = self.flame(
                cameras=self.cam2world[..., :3, :3],
                shape_params=shape.repeat(bs, 1),
                expression_params=exp.repeat(bs, 1),
                eye_pose_params=eyes.repeat(bs, 1),
                jaw_pose_params=jaw.repeat(bs, 1),
                eyelid_params=eyelids.repeat(bs, 1),
                rot_params=R.repeat(bs, 1),
                trans_params=t.repeat(bs, 1),
                scale_params=scale,
            )
            right_eye, left_eye = eyes[:, :6], eyes[:, 6:]
            proj_lmksMP = self.transform_points_screen(lmkMP)[..., :2]
            proj_lmks68 = self.transform_points_screen(lmk68)[..., :2]
            proj_vertices = self.transform_points_screen(vertices)[..., :2]
            recon = self.render_img(vertices, texture)

            losses = {}
            # Landmarks sparse term
            image_size = (self.height, self.width)
            losses['loss/lmk_oval'] = util.oval_lmk_loss(proj_lmks68, image_lmks68, image_size, lmk_mask) * self.config.w_lmks_oval
            losses['loss/lmk_MP'] = util.face_lmk_loss(proj_lmksMP, image_lmksMP, image_size, True, lmk_dense_mask) * self.config.w_lmks
            losses['loss/lmk_eye'] = util.eye_closure_lmk_loss(proj_lmksMP, image_lmksMP, image_size, lmk_dense_mask) * self.config.w_lmks_lid
            losses['loss/lmk_mouth'] = util.mouth_lmk_loss(proj_lmksMP, image_lmksMP, image_size, True, lmk_dense_mask) * self.config.w_lmks_mouth
            losses['loss/lmk_iris_left'] = util.lmk_loss(proj_vertices[:, left_iris_flame, ...], left_iris, image_size, mask_left_iris) * self.config.w_lmks_iris
            losses['loss/lmk_iris_right'] = util.lmk_loss(proj_vertices[:, right_iris_flame, ...], right_iris, image_size, mask_right_iris) * self.config.w_lmks_iris

            # Photometric term
            losses["loss/pho"] = F.l1_loss(recon, self.images ** self.gamma) * self.config.w_pho

            # Reguralizers
            losses['reg/exp'] = torch.sum(exp ** 2) * self.config.w_exp
            losses['reg/tex'] = torch.sum(texture ** 2) * self.config.w_tex
            losses['reg/shape'] = torch.sum((shape - self.mica_shape) ** 2) * self.config.w_shape
            losses['reg/sym'] = torch.sum((right_eye - left_eye) ** 2) * 8.0
            losses['reg/jaw'] = torch.sum((I6D - jaw) ** 2) * 16.0
            losses['reg/eye_lids'] = torch.sum((eyelids[:, 0] - eyelids[:, 1]) ** 2)
            losses['reg/eye_left'] = torch.sum((I6D - left_eye) ** 2)
            losses['reg/eye_right'] = torch.sum((I6D - right_eye) ** 2)

            all_loss = self.reduce_loss(losses)
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            scheduler.step()

            for key in losses.keys():
                self.writer.add_scalar(key, losses[key], global_step=p)
            
            if (p + 1) % log_freq == 0:
                save_image(
                    torch.cat([recon ** (1 / self.gamma), self.images], dim=0), 
                    os.path.join(self.save_root, "vis_%05d.png" % (p + 1)), 
                    nrow=len(recon)
                )
        
        trimesh.Trimesh(
            vertices=vertices[0].detach().cpu().numpy(),
            faces=self.flame.faces.detach().cpu().numpy(),
        ).export(os.path.join(self.save_root, "flame_mesh.obj"))

        res = {
            "shape": shape,
            "texture": texture,
            "exp": exp,
            "eyes": eyes,
            "eyelids": eyelids,
            "jaw": jaw,
            "t": t,
            "R": R,
            "scale": scale,
            "intrinsic": self.intrinsic,
            "act_trans": act_trans,
        }
        torch.save(res, os.path.join(self.save_root, "fitting.pkl"))

        return res

    def render_specular(self, res):
        meta_file_path = os.path.join(self.data_root, "transforms.json")
        with open(meta_file_path, 'r') as f:
            meta = json.load(f)
        
        # load fitting model
        shape = res['shape']
        texture = res['texture']
        exp = res['exp']
        eyes = res['eyes']
        eyelids = res['eyelids']
        jaw = res['jaw']
        t = res['t']
        R = res['R']
        scale = res['scale']
        vertices, lmk68, lmkMP, _ = self.flame(
            cameras=torch.eye(3).to(self.device)[None, ...],
            shape_params=shape,
            expression_params=exp,
            eye_pose_params=eyes,
            jaw_pose_params=jaw,
            eyelid_params=eyelids,
            rot_params=R,
            trans_params=t,
            scale_params=scale,
        )
            
        # load extrinsics, images, and landmarks
        frames = meta["frames"]
        num_frames = len(frames)
        spec_save_root = os.path.join(self.data_root, "specular_bfm")
        inpaint_save_root = os.path.join(self.data_root, "inpaint_mask")
        os.makedirs(spec_save_root, exist_ok=True)
        os.makedirs(inpaint_save_root, exist_ok=True)
        for f_id in tqdm(range(num_frames)):
            img_name = os.path.basename(frames[f_id]['file_path'])
            cur_pose = np.array(frames[f_id]['transform_matrix'], dtype=np.float32)
            cur_pose = nerf_matrix_to_ngp(cur_pose, scale=1.)
            cam2world = torch.from_numpy(cur_pose).to(self.device)
            extrin = torch.inverse(cam2world)
            diff_albedo, spec_albedo = self.flametex(texture)  # [1,3,h,w]
            normal = util.vertex_normals(vertices, self.faces[:1])  # [1,v,3]
            attr = torch.zeros_like(vertices)
            mesh_dict = {
                "vertice": vertices,
                "attributes": normal,
                "faces": self.faces[:1],  # [1,f,3]
                "face_attributes": self.face_uvcoords[None, ...],  # [1,f,3,2]
                "size": (self.height, self.width),
            }
            cam_int = torch.clone(self.intrinsic)
            cam_int[0] /= self.width
            cam_int[1] /= self.height
            _, viewdirs = _compute_rays(
                cam2world[None, ...], self.intrinsic[None, ...], self.height, self.width, self.device
            )
            viewdirs = viewdirs.permute(0, 3, 1, 2)  # [b,3,h,w]
            
            normal_img, uv, pix_to_face = self.mesh_renderer.render_mesh(
                mesh_dict, cam_int[None, ...], extrin[None, :3],
            )
            frontal_mask = (torch.sum(-normal_img * viewdirs, dim=1, keepdim=True).float() > 0).float()
            uv = uv.permute(0, 2, 3, 1)  # [1,h,w,2]
            uv = 2 * uv - 1
            uv[..., 1] *= -1
            spec_albedo_img = F.grid_sample(spec_albedo, uv)
            face_mask = F.grid_sample(self.uv_face_mask.repeat(len(vertices), 1, 1, 1), uv)
            eye_mask = F.grid_sample(self.uv_eye_mask.repeat(len(vertices), 1, 1, 1), uv)
            save_image(spec_albedo_img * face_mask * frontal_mask, os.path.join(spec_save_root, img_name))
            save_image(1. - face_mask + eye_mask, os.path.join(inpaint_save_root, img_name))


if __name__ == "__main__":
    config = parse_args()
    flame_fitter = FlameFitter(
        config=config,
        device="cuda",
    )
    res = flame_fitter.optimize_landmarks(iters=1000)
    flame_fitter.render_specular(res)
