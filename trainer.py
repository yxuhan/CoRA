import os
import argparse
import yaml
import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from nerfacc import OccupancyGrid
import lpips
import kornia
import trimesh
import pymeshlab
import pymeshfix
from pytorch3d.utils import ico_sphere

from cora.extract_geometry import extract_geometry
from cora.mesh_renderer import MeshRenderer
from cora.dataset import MetaShapeDataset as AvatarDataset
from cora.renderer import ObjSDFRenderer as Renderer
from cora.renderer import Rays
from cora.model import NeuSAvatar as Network


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config_path', type=str)
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument(
    '--mode', type=str, default="train", choices=[
        "train", "export_eyeball",
    ]
)

# these options can overwrite the same options in configs if specified
parser.add_argument('--device', type=str)
parser.add_argument('--chunk_size', type=int)

# save path configs
parser.add_argument('--save_visual_dir', type=str, default="workspace/visual")

opt, _ = parser.parse_known_args()


os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
opt.device = "cuda"


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        with open(opt.config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        if opt.device is not None: cfg["device"] = opt.device
        if opt.chunk_size is not None: cfg["renderer"]["chunk_size"] = opt.chunk_size

        self.device = self.cfg["device"]
        
        # metrics
        self.lpips_loss = lpips.LPIPS(net="vgg").to(self.device)
        self.lpips_loss.requires_grad_(False)
        self.ssim_loss = kornia.metrics.ssim
        self.psnr_loss = kornia.metrics.psnr

        self._create_dataset()
        self._create_model()
        self._create_renderer()

    def _create_dataset(self):
        cfg = self.cfg
                        
        if opt.mode in ["train"]:
            self.train_dataset = AvatarDataset(cfg=cfg, mode="train")
            self.val_dataset = AvatarDataset(cfg=cfg, mode="val")
            self.HEIGHT = self.train_dataset.HEIGHT
            self.WIDTH = self.train_dataset.WIDTH
        else:
            self.test_exp_dataset = AvatarDataset(cfg=cfg, mode=opt.mode)
            self.HEIGHT = self.test_exp_dataset.HEIGHT
            self.WIDTH = self.test_exp_dataset.WIDTH

    def _create_model(self):
        cfg = self.cfg
        opt = self.opt
        self.bound = self.cfg["data"]["bound"]
        self.radiance_field = Network(cfg)
        self.radiance_field.to(self.device)

    def _create_renderer(self):
        opt = self.opt
        cfg = self.cfg
        
        self.device = cfg["device"]
        bound = cfg["data"]["bound"]
        scene_aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound]).to(self.device)
        self.bound = bound
        self.render_step_size = (
            (scene_aabb[3:] - scene_aabb[:3]).max()
            * math.sqrt(3)
            / cfg["renderer"]["render_n_samples"]
        ).item()
        self.occupancy_grid = OccupancyGrid(
            roi_aabb=scene_aabb,
        ).to(self.device)

        self.renderer = Renderer(
            radiance_field=self.radiance_field,
            occupancy_grid=self.occupancy_grid,
            scene_aabb=scene_aabb,
        )
        self.mesh_renderer = MeshRenderer(device=self.device)

    def _compute_rays(self, c2w, x, y, attr, intrinsic, reshape=False):
        '''
        x, y: [num_rays]
        attr: [h,w,c]
        c2w: [4,4] or [3,4] (after refine)
        '''
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

        if reshape:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            for k in attr:
                try:
                    attr[k] = attr[k].reshape(self.HEIGHT, self.WIDTH, -1)
                except:
                    pass
        return attr, origins, viewdirs

    def _process_data(self, data, test=False):
        data["attr"] = data["rand_attr"]
        x, y = data["rand_x"], data["rand_y"]
        
        intrinsic = data["intrinsic"]
        c2w = data["attr"]["c2w"]

        attr = data["attr"]  # [nray,4]
        point_attrs, origins, viewdirs = self._compute_rays(c2w, x, y, attr, intrinsic, reshape=test)
        data["origins"] = origins
        data["viewdirs"] = viewdirs
        data["rays"] = Rays(data["origins"], data["viewdirs"])
        for k in point_attrs:
            data[k] = point_attrs[k]
                
        return data

    def _create_optimizer(self):
        params = [
            {"params": self.radiance_field.parameters(), "lr": 1e-3, "eps": 1e-15},
            # {"params": self.camera_offset.parameters(), "lr": 1e-3, "eps": 1e-15},
        ]
        self.optimizer = torch.optim.Adam(params=params)
        if self.cfg["train"]["warmup_steps"] > 0:
            linear_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.01, end_factor=1., total_iters=self.cfg["train"]["warmup_steps"]
            )
            step_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.cfg["train"]["milestones"],
                gamma=0.33,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                [linear_scheduler, step_scheduler],
                milestones=[self.cfg["train"]["warmup_steps"]],
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.cfg["train"]["milestones"],
                gamma=0.33,
            )

    def train(self):
        cfg = self.cfg

        # create workspace
        workspace = os.path.join(cfg["workspace"], cfg["name"])
        os.makedirs(workspace, exist_ok=True)
        os.system("cp %s %s" % (opt.config_path, os.path.join(workspace, "config.yaml")))
        writer = SummaryWriter(workspace)

        # create optimizer
        self.grad_scaler = torch.cuda.amp.GradScaler(2 ** 10)
        self._create_optimizer()
        
        max_steps = cfg["train"]["max_steps"]
        self.step = 0
        if self.opt.ckpt_path is not None:
            self._load_checkpoints(is_train=True)
        
        export_material = False
        while True:
            if self.step > max_steps:
                print("Training complete, saving checkpoints...")
                self._save_checkpoints(os.path.join(workspace, "latest.pth"))
                exit()

            # set control flags
            # if use mesh renderer, geometry is fixed, only optimize the textures
            self.mesh_render = self.step > cfg["train"]["mesh_render_iter"] and cfg["train"]["mesh_render"]
            if self.mesh_render and not export_material:
                self.train_dataset.set_mesh_render_mode()
                precompute_mesh_path = os.path.join(workspace, "precompute_mesh_%05d.obj" % self.step)
                self._clip_mesh_fn(self.train_dataset, precompute_mesh_path)
                precompute_mesh = trimesh.load_mesh(precompute_mesh_path)
                self.precompute_vertices = torch.from_numpy(precompute_mesh.vertices).float().to(self.device)[None, ...]
                self.precompute_faces = torch.from_numpy(precompute_mesh.faces).to(self.device)[None, ...]
                export_material = True

            # load data
            dataset = self.train_dataset
            train_data = dataset[np.random.randint(0, len(dataset))]
            self.train_data = self._process_data(train_data, test=False)

            # compute loss
            self.radiance_field.train()
            self.radiance_field.update_step(self)
            self.occupancy_grid.every_n_step(
                step=self.step,
                occ_eval_fn=lambda x: self.radiance_field.query_opacity(x, self.render_step_size),
            )
            if self.mesh_render:
                loss, loss_dict = self.radiance_field.train_mesh_step_precompute(self)
            else:
                loss, loss_dict = self.radiance_field.train_step(self, ignore_eyeball=False)
            
            if loss_dict is None:
                continue
            if self.step % 10 == 0:
                for k in loss_dict:
                    writer.add_scalar('Loss/%s' % k, loss_dict[k].item(), self.step)
            self.step += 1
            
            # update parameters
            self.optimizer.zero_grad()
            # with autograd.detect_anomaly():
            self.grad_scaler.scale(loss).backward()
            if self.mesh_render:
                for name, params in self.radiance_field.named_parameters():
                    if "material" not in name:
                        params.grad = None
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.scheduler.step()

            if self.step % cfg["train"]["save_freq"] == 0:
                save_path = os.path.join(workspace, "iter_%05d.pth" % self.step)
                print("save checkpoints to %s..." % save_path)
                self._save_checkpoints(save_path)
            if self.step % 1000 == 0:
                self._save_checkpoints(os.path.join(workspace, "latest.pth"))

            # validation
            if self.step % cfg["train"]["val_freq"] == 0:
                if self.step % cfg["train"]["vis_freq"] == 0:
                    save_visual = True
                    val_dir = os.path.join(workspace, "iter_%08d" % self.step)
                    os.makedirs(val_dir, exist_ok=True)
                else:
                    save_visual = False
                    val_dir = None
                self.radiance_field.eval()
                with torch.no_grad():
                    print("Validating...")
                    val_dataset = self.val_dataset
                    self._valid_loop(
                        dataset=val_dataset,
                        prefix="inv-render",
                        save_visual=save_visual,
                        save_visual_dir=val_dir,
                        writer=writer,
                        inv_render=True,
                        mesh_render=self.mesh_render,
                    )

    def _save_checkpoints(self, path):
        weight_dict = {
            "radiance_field": self.radiance_field.state_dict(),
            "occupancy_grid": self.occupancy_grid.state_dict(),
            # "camera_offset": self.camera_offset.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.grad_scaler.state_dict(),
            "step": self.step,
        }
        torch.save(weight_dict, path)
    
    def _load_checkpoints(self, is_train=False):
        weight_dict = torch.load(self.opt.ckpt_path, map_location=self.device)
        print("loading from %s..." % self.opt.ckpt_path)
        self.radiance_field.load_state_dict(weight_dict["radiance_field"])
        # self.camera_offset.load_state_dict(weight_dict["camera_offset"])
        self.occupancy_grid.load_state_dict(weight_dict["occupancy_grid"])
        if is_train:
            self.optimizer.load_state_dict(weight_dict["optimizer"])
            self.scheduler.load_state_dict(weight_dict["scheduler"])
            self.grad_scaler.load_state_dict(weight_dict["scaler"])
            self.step = weight_dict["step"]
            print("Continue training from %d steps" % self.step)

    def _valid_loop(self, dataset, prefix, save_visual, save_visual_dir, writer=None, inv_render=False, mesh_render=False):
        metric_dict = {}
        for i in range(len(dataset)):
            val_data = dataset[i]
            self.val_data = self._process_data(val_data, test=True)
            if mesh_render:
                res_dict = self.radiance_field.valid_mesh_step(self)
            else:  # volume rendering
                res_dict = self.radiance_field.valid_step(self, inv_render=inv_render)
            if save_visual:
                save_image(res_dict["vis"], os.path.join(save_visual_dir, "%s_%05d.png" % (prefix, i)))
            # merge metric
            for k in ["SSIM", "PSNR", "LPIPS"]:
                cur_val = res_dict[k]
                if k in metric_dict:
                    metric_dict[k].append(cur_val)
                else:
                    metric_dict[k] = [cur_val]
        for k in metric_dict:
            cur_metric = sum(metric_dict[k]) / len(metric_dict[k])
            print("[%s][%d steps]: %s %.4f" % (
                prefix, self.step, k, cur_metric,
            ))
            if writer is not None:
                writer.add_scalar('%s/%s' % (prefix, k), cur_metric, self.step)

    def _clip_mesh_fn(self, dataset, mesh_save_path, ignore_eyeball=False):
        '''
        render the mesh for each view, and then delete the unseen faces
        '''
        extract_mesh = False
        if "volsdf" in self.cfg["backbone"]:
            query_func = lambda pts: -self.radiance_field.query_geometry(pts, ignore_eyeball) + self.radiance_field.beta()
        else:
            query_func = lambda pts: -self.radiance_field.query_geometry(pts, ignore_eyeball)
        
        with torch.no_grad():
            for i in tqdm(range(len(dataset))):
                val_data = dataset[i]
                val_data = self._process_data(val_data, test=True)
                self.val_data = val_data

                if not extract_mesh:
                    # marching cube in the SDF field to extract initial mesh
                    print("extracting mesh ...")
                    vertices, triangles = extract_geometry(
                        bound_min=torch.tensor([-self.bound, -self.bound, -self.bound]),
                        bound_max=torch.tensor([self.bound, self.bound, self.bound]),
                        resolution=256,
                        threshold=0,
                        query_func=query_func,
                        device=self.device,
                    )  # [v,3] [f,3]
                    mesh = trimesh.Trimesh(vertices, triangles)
                    mesh.export(mesh_save_path)
                    # mesh simplification
                    print("mesh simplification ...")
                    ms = pymeshlab.MeshSet()
                    ms.load_new_mesh(mesh_save_path)
                    ms.meshing_decimation_quadric_edge_collapse(targetperc=0.5)
                    ms.save_current_mesh(mesh_save_path)
                    mesh = trimesh.load_mesh(mesh_save_path)
                    vertices, triangles = mesh.vertices, mesh.faces
                    extract_mesh = True
                    print("select visible faces on the mesh for further simplification ...")
                    vert = torch.from_numpy(vertices).float().to(self.device)
                    tri = torch.from_numpy(triangles.astype(np.int64)).to(self.device)
                    mesh_dict = {
                        "vertice": vert[None, ...],
                        "faces": tri[None, ...],
                        "attributes": torch.ones_like(vert[None, ...]),
                        "size": (self.HEIGHT, self.WIDTH),
                    }
                    seen_mask = torch.zeros_like(tri[..., 0])

                # rasterization to find visible faces
                cam_int = torch.clone(val_data["intrinsic"])  # [3,3]
                cam_int[0] /= self.WIDTH
                cam_int[1] /= self.HEIGHT
                cam_int = cam_int[None, ...]  # [1,3,3]
                cam_ext = torch.inverse(val_data["attr"]["c2w"][0])[:-1][None, ...]  # [1,3,4]
                output, pix_to_face = self.mesh_renderer.render_mesh(mesh_dict, cam_int, cam_ext)  # [1,3,h,w] [1,h,w,1]

                # render the neural fields to find visible mask
                if ignore_eyeball:
                    vis_mask = val_data["mask"] - val_data["leye_mask"] - val_data["reye_mask"]
                else:
                    vis_mask = val_data["mask"]
                # res_dict = self.radiance_field.compute_mask_step(self, ignore_eyeball=ignore_eyeball)
                # vis_mask = res_dict["mask"]  # [1,1,h,w]
                
                # select visible faces in the current viewpoint
                pix_to_face = pix_to_face.reshape(-1, 1)  # [hw,1]
                vis_mask = vis_mask.reshape(-1, 1)  # [hw,1]
                cur_seen_face = pix_to_face[torch.logical_and(pix_to_face > -1, vis_mask > 0.5)]
                seen_mask[cur_seen_face] = 1

            seen_mask = seen_mask.cpu().numpy()
            mesh.update_faces(seen_mask > 0)

            # only save the connection part with the largest num of faces
            split_all = mesh.split(only_watertight=False)
            mesh = sorted(split_all, key=lambda x: len(x.faces))[-1]
            vertices, triangles = mesh.vertices, mesh.faces
            tin = pymeshfix.PyTMesh()
            tin.load_array(vertices, triangles)
            tin.fill_small_boundaries(nbe=100)
            vertices, triangles = tin.return_arrays()
            # vertices, triangles = pymeshfix.clean_from_arrays(vertices, triangles)
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(mesh_save_path)

    def _export_matrial_fn(self, export_dir, dataset, ignore_eyeball=False, uv_size=4096):
        os.makedirs(export_dir, exist_ok=True)
        mesh_render = MeshRenderer(device=self.device)
        mesh_save_path = os.path.join(export_dir, "mesh.obj")
        mesh_uv_save_path = os.path.join(export_dir, "mesh_uv.obj")
        diff_save_path = os.path.join(export_dir, "diffuse.png")
        normal_save_path = os.path.join(export_dir, "normal.png")
        spec_save_path = os.path.join(export_dir, "specular.png")
        rough_save_path = os.path.join(export_dir, "roughness.png")
        coord_save_path = os.path.join(export_dir, "coord.pkl")
        with torch.no_grad():
            self._clip_mesh_fn(dataset, mesh_save_path, ignore_eyeball)
            print("UV unwarping ...")
            
            # Blender UV
            blender_path = "blender/blender-3.1.0-linux-x64/blender"
            os.system(
                "%s --background --python blender/export_blender.py %s %s" % (
                    blender_path, mesh_save_path, mesh_uv_save_path,
                )
            )
            mesh = trimesh.load_mesh(mesh_uv_save_path)
            uv = torch.from_numpy(mesh.visual.uv).to(self.device).float()  # [v,2]
            vertices = torch.from_numpy(mesh.vertices).to(self.device).float()  # [v,3]
            faces = torch.from_numpy(mesh.faces).to(self.device)
            uv = 2 * uv - 1  # [0,1] to [-1,1]            
            uv[..., 0] *= -1  # in pytorch3d, y to up, x to left, in uv, y to up, x to right
            uvz = torch.cat([uv, torch.ones_like(uv[..., -1:])], dim=-1)  # [v,3]
            mesh_dict = {
                "faces": faces[None, ...],
                "vertice": uvz[None, ...],
                "attributes": vertices[None, ...],
                "size": (uv_size, uv_size),
            }
            coords, _ = mesh_render.render_ndc(mesh_dict)  # [1,3,h,w]
            torch.save(coords, coord_save_path)
            coords = coords[0].permute(1, 2, 0)  # [h,w,3]
            
            h, w = coords.shape[:-1]  # uh, uw
            coords = coords.reshape(-1, 3).to(self.device)
            chunk_size = self.cfg["renderer"]["chunk_size"]
            left = 0
            diff_list, specular_list, roughness_list, normal_list = [], [], [], []
            while True:
                cur_coords = coords[left:left + chunk_size]
                out = self.radiance_field.query_attributes(cur_coords, return_brdf=True, return_normal=True, detach_normal=True)
                diff = out["diff"]
                specular = out["spec"]["specular"]
                roughness = out["spec"]["roughness"]
                normal = out["normal"]
                diff_list.append(diff.detach().cpu())
                specular_list.append(specular.detach().cpu())
                roughness_list.append(roughness.detach().cpu())
                normal_list.append(normal.detach().cpu())
                left = left + chunk_size
                if left >= h * w:
                    break
            diff_list = torch.cat(diff_list, dim=0).reshape(h, w, -1).permute(2, 0, 1)
            specular_list = torch.cat(specular_list, dim=0).reshape(h, w, -1).permute(2, 0, 1)
            roughness_list = torch.cat(roughness_list, dim=0).reshape(h, w, -1).permute(2, 0, 1)
            normal_list = torch.cat(normal_list, dim=0).reshape(h, w, -1).permute(2, 0, 1)
            save_image(diff_list, diff_save_path)
            save_image(specular_list, spec_save_path)
            save_image(roughness_list, rough_save_path)
            save_image((normal_list + 1) / 2, normal_save_path)
            # post_process(export_dir)

    def _export_eyeball_material_fn(self, export_dir, pos, radius):
        os.makedirs(export_dir, exist_ok=True)
        mesh_render = MeshRenderer(device=self.device)
        mesh_save_path = os.path.join(export_dir, "mesh.obj")
        mesh_uv_save_path = os.path.join(export_dir, "mesh_uv.obj")
        diff_save_path = os.path.join(export_dir, "diffuse.png")
        normal_save_path = os.path.join(export_dir, "normal.png")
        coord_save_path = os.path.join(export_dir, "coord.pkl")

        if "volsdf" in self.cfg["backbone"]:
            radius = radius + self.radiance_field.beta()

        mesh_render = MeshRenderer(device=self.device)
        mesh = ico_sphere(level=3, device=self.device)
        verts = mesh.verts_list()[0]  # [v,3]
        faces = mesh.faces_list()[0]  # [f,3]
        verts = verts * radius + pos
        mesh_np = trimesh.Trimesh(
            vertices=verts.detach().cpu().numpy(),
            faces=faces.detach().cpu().numpy(),
        )
        seen_mask = torch.ones_like(faces[..., 0])
        for i in tqdm(range(len(self.test_exp_dataset))):
            tmp_data = self.test_exp_dataset[i]
            tmp_data = self._process_data(tmp_data, test=True)
            view_mean = torch.mean(tmp_data["viewdirs"], dim=(0, 1))  # [3]
            view_mean = F.normalize(view_mean, dim=-1)
            face_normal = torch.from_numpy(mesh_np.face_normals).to(self.device)
            cos = torch.sum(face_normal * view_mean, dim=-1)
            seen_mask[cos > 0.8] = 0
        mesh_np.update_faces(seen_mask.cpu().numpy() > 0)
        split_all = mesh_np.split(only_watertight=False)
        mesh_np = sorted(split_all, key=lambda x: len(x.faces))[-1]
        mesh_np.export(mesh_save_path)

        # Blender UV
        blender_path = "blender/blender-3.1.0-linux-x64/blender"
        os.system(
            "%s --background --python blender/export_blender.py %s %s" % (
                blender_path, mesh_save_path, mesh_uv_save_path,
            )
        )

        mesh = trimesh.load_mesh(mesh_uv_save_path)
        uv = torch.from_numpy(mesh.visual.uv).to(self.device).float()  # [v,2]
        vertices = torch.from_numpy(mesh.vertices).to(self.device).float()  # [v,3]
        faces = torch.from_numpy(mesh.faces).to(self.device)
        uv = 2 * uv - 1  # [0,1] to [-1,1]            
        uv[..., 0] *= -1  # in pytorch3d, y to up, x to left, in uv, y to up, x to right
        uvz = torch.cat([uv, torch.ones_like(uv[..., -1:])], dim=-1)  # [v,3]
        mesh_dict = {
            "faces": faces[None, ...],
            "vertice": uvz[None, ...],
            "attributes": vertices[None, ...],
            "size": (128, 128),
        }
        coords, _ = mesh_render.render_ndc(mesh_dict)  # [1,3,h,w]
        torch.save(coords, coord_save_path)
        coords = coords[0].permute(1, 2, 0)  # [h,w,3]
        h, w = coords.shape[:-1]  # uh, uw
        coords = coords.reshape(-1, 3).to(self.device)
        chunk_size = self.cfg["renderer"]["chunk_size"]
        left = 0
        diff_list, normal_list = [], []
        while True:
            cur_coords = coords[left:left + chunk_size]
            with torch.no_grad():
                out = self.radiance_field.query_attributes(
                    cur_coords, return_brdf=True, return_normal=False
                )
            diff = out["diff"]
            normal = F.normalize(cur_coords - pos, dim=-1)
            diff_list.append(diff)
            normal_list.append(normal)
            left = left + chunk_size
            if left >= h * w:
                break
        diff_list = torch.cat(diff_list, dim=0).reshape(h, w, -1).permute(2, 0, 1)
        normal_list = torch.cat(normal_list, dim=0).reshape(h, w, -1).permute(2, 0, 1)
        save_image(diff_list, diff_save_path)
        save_image((normal_list + 1) / 2, normal_save_path)
        # post_process(export_dir)
    
    def export_eyeball_material(self):
        self._load_checkpoints()
        self.radiance_field.eval()
        self._export_matrial_fn(
            export_dir=self.opt.save_visual_dir,
            dataset=self.test_exp_dataset,
            ignore_eyeball=True,
        )
        eyeball_size = self.radiance_field.eyeball.get_world_eyeball_size()
        leye_center, reye_center = self.radiance_field.eyeball.get_world_eyeball_center()
        self._export_eyeball_material_fn(
            export_dir=os.path.join(self.opt.save_visual_dir, "right"),
            pos=leye_center,
            radius=eyeball_size,
        )
        self._export_eyeball_material_fn(
            export_dir=os.path.join(self.opt.save_visual_dir, "left"),
            pos=reye_center,
            radius=eyeball_size,
        )    


if __name__ == "__main__":
    trainer = Trainer(opt)
    if opt.mode == "train":
        trainer.train()
    elif opt.mode == "export_eyeball":
        trainer.export_eyeball_material()
    else:
        raise NotImplementedError
