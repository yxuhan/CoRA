import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import trimesh

from .networks import (
    BetaNetwork, VolumeMaterial, VolumeRadianceHead, VolumeSDF, Shader, EyeballSDF
)


class NeuSAvatar(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bound = cfg["data"]["bound"]
        self.num_levels = cfg["network"]["num_levels"]
        self.level_dim = cfg["network"]["level_dim"]
        self.neighbor_eps = cfg["network"]["neighbor_eps"]
        # self.lr_scale = 10.
        self.light_scale = torch.nn.parameter.Parameter(torch.tensor([1.]))
        self.light_intensity = torch.tensor(cfg["light"]["color"]) * cfg["light"]["intensity"]
        self.gamma = cfg["light"]["gamma"]
        self.device = cfg["device"]
        self.level_mask = torch.ones(self.num_levels * self.level_dim)
        
        self._set_networks()

    def _set_networks(self):
        num_levels = self.num_levels
        level_dim = self.level_dim
        per_level_scale = np.exp2(np.log2(1024 / 16) / (num_levels - 1))
        geo_feat_dim = self.cfg["network"]["geo_feat_dim"]
        self.geometry = VolumeSDF(
            num_levels=num_levels,
            level_dim=level_dim,
            per_level_scale=per_level_scale,
            n_output_dims=geo_feat_dim,
        )
        self.material = VolumeMaterial(
            num_levels=num_levels,
            level_dim=level_dim,
            per_level_scale=per_level_scale,
            brdf_cfg=self.cfg["brdf"],
        )
        self.shader = Shader(
            brdf_cfg=self.cfg["brdf"],
        )
        self.beta = BetaNetwork()
        self.texture = VolumeRadianceHead(
            input_feature_dim=geo_feat_dim,
        )
        self.eyeball = EyeballSDF(cfg=self.cfg["data"])

    def forward_geometry(self, x):
        '''
        return sdf and feature
        '''
        return self.geometry(x)

    def forward_material(self, x):
        '''
        return diff and spec
        '''
        return self.material(x)
        
    def forward(self, x, view_dir, view_pos, dist2, c2w, light_pos=None, only_render_mask=False, 
                ignore_eyeball=False, **kwargs):
        if only_render_mask:
            out = self.query_attributes(
                x, return_brdf=False, return_normal=False, detach_normal=True, ignore_eyeball=ignore_eyeball,
            )
            density = out["density"]
            return torch.zeros_like(density), density, torch.zeros_like(x)
        
        if len(x) == 0:
            n_attr_chns = 20 + 4
            return (
                torch.zeros(*x[..., 0].size(), n_attr_chns).to(x.device),  # dummy attrs
                torch.zeros_like(x[..., :1]),  # dummy density
                torch.zeros_like(x),
            )
        out = self.query_attributes(
            x, return_brdf=True, return_normal=True, ignore_eyeball=ignore_eyeball,
        )
        density, density_obj = out["density"], out["density_obj"]
        normal, grad = out["normal"], out["grad"]
        diff_color, spec_params = out["diff"], out["spec"]
        feature = out["feature"]
        sdf, fsdf, lsdf, rsdf = out["sdf"], out["face_sdf"], out["leye_sdf"], out["reye_sdf"]

        color = self.texture(feature, view_dir, normal)
        
        diff_shading, spec_shading = self.shader.shading(
            pos_world=x, diff_color=diff_color, spec_params=spec_params, normal=normal, view_pos_world=view_pos,
            light_pos_world=light_pos, c2w=c2w,
        )
        diff_shading = diff_shading * self.light_intensity.to(x.device)
        spec_shading = spec_shading * self.light_intensity.to(x.device)
        diff_shading = diff_shading / dist2
        spec_shading = spec_shading / dist2
        
        attrs = torch.cat([
            color, normal, diff_shading, spec_shading, grad, spec_params["specular"], sdf, 
            fsdf, rsdf, lsdf,
        ], dim=-1)
        
        with torch.no_grad():
            rand_dir = torch.randn_like(normal)
            rand_dir = F.normalize(rand_dir, dim=-1)
            pert_dir = torch.cross(rand_dir, normal)
            x_eps = x + self.neighbor_eps * pert_dir * torch.randn_like(normal[..., -1:])
        out = self.query_attributes(
            x_eps, return_brdf=True, return_normal=True, ignore_eyeball=ignore_eyeball,
        )
        normal_eps, spec_eps = out["normal"], out["spec"]["specular"]
        attrs = torch.cat([attrs, normal_eps, spec_eps], dim=-1)
        
        return attrs, density, density_obj

    def query_attributes(self, x, return_brdf=False, return_normal=False, detach_normal=False, 
                         ignore_eyeball=False):
        if len(x) == 0:  # avoid 0 samples in ray during ray marching
            return {"density": torch.zeros_like(x[..., :1])}
        
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        x.requires_grad_(return_normal)
        out = {}
        with torch.set_grad_enabled(return_normal):
            beta = self.beta()
            sdf, feature = self.forward_geometry(x)
            leye_sdf, reye_sdf = self.eyeball(x)
            if ignore_eyeball:
                real_sdf = sdf
            else:
                real_sdf = torch.minimum(reye_sdf, leye_sdf)
                real_sdf = torch.minimum(real_sdf, sdf)
                eye_mask = (real_sdf < sdf).float()
            
            density = sdf_to_density(real_sdf, beta)
            density_obj = torch.cat([
                sdf_to_density(sdf, beta), sdf_to_density(leye_sdf, beta), sdf_to_density(reye_sdf, beta),
            ], dim=-1)
            out.update({
                "density": density,
                "density_obj": density_obj,
                "sdf": real_sdf,
                "feature": feature,
                "face_sdf": sdf,
                "leye_sdf": leye_sdf,
                "reye_sdf": reye_sdf,
            })
            if return_normal:
                grad = torch.autograd.grad(
                    real_sdf,
                    x,
                    grad_outputs=torch.ones_like(sdf),
                    create_graph=not detach_normal, 
                    retain_graph=not detach_normal, 
                )[0]
                normal = F.normalize(grad, dim=-1)
                out.update({
                    "grad": grad,
                    "normal": normal,
                })
            if return_brdf:
                diff, spec = self.forward_material(x)
                spec["specular"] = spec["specular"] * (1 - eye_mask) + eye_mask * 0.2
                spec["roughness"] = spec["roughness"] * (1 - eye_mask) + eye_mask * 0.05
                out.update({
                    "diff": diff,
                    "spec": spec,
                })
        return out

    def query_geometry(self, x, ignore_eyeball=False):
        out = self.query_attributes(
            x, return_brdf=False, return_normal=False, detach_normal=True, ignore_eyeball=ignore_eyeball,
        )
        return out["sdf"]
    
    def query_density(self, x, **kwargs):
        out = self.query_attributes(x, return_brdf=False, return_normal=False, detach_normal=True)
        return out["density"]

    def query_opacity(self, x, step_size):
        out = self.query_attributes(x, return_brdf=False, return_normal=False)
        density = out["density"]
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    def get_attr_dict(self, attrs):
        attr_dict = {
            "color": attrs[..., :3],
            "normal": attrs[..., 3:6],
            "diff_shading": attrs[..., 6:9],
            "spec_shading": attrs[..., 9:12],
            "sdf_grad": attrs[..., 12:15],
            "specular": attrs[..., 15:16],
            "sdf": attrs[..., 16:17],
            "face_sdf": attrs[..., 17:18],
            "reye_sdf": attrs[..., 18:19],
            "leye_sdf": attrs[..., 19:20],
            "eps_normal": attrs[..., 20:23],
            "eps_specular": attrs[..., 23:24],
        }
        return attr_dict

    def update_step(self, trainer):
        cfg = self.cfg["train"]
        step = trainer.step
        geometry_level = cfg["init_level"] + step // cfg["level_grow_iter"]                
        self.geometry.update_level(geometry_level)

    def train_step(self, trainer, ignore_eyeball):
        # parse params
        cfg = self.cfg
        data = trainer.train_data
        self.batch = data
        renderer = trainer.renderer
        render_step_size = trainer.render_step_size
        sample_patch = trainer.sample_patch

        ########
        # psrse data
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]
        pixels = pixels ** self.gamma
        mask = data["mask"]
        leye_mask = data["leye_mask"]
        reye_mask = data["reye_mask"]
        
        ########
        # network forward
        attr, acc, depth, n_rendering_samples, sample_weights, sample_attrs, ray_idx = renderer.render_image(
            rays,
            # rendering options
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            # other kwargs
            c2w=data["c2w"],
            ignore_eyeball=ignore_eyeball,
        )
        if n_rendering_samples == 0:
            return None, None
        
        ########
        # psrse render results
        num_rays = len(pixels)
        attr_dict = self.get_attr_dict(attr)
        sample_attr_dict = self.get_attr_dict(sample_attrs)
        diff_shading = attr_dict["diff_shading"]
        spec_shading = attr_dict["spec_shading"]
        pred = attr_dict["color"]
        recon = diff_shading + spec_shading
        acc_scene = acc[..., :1]
        acc_face = acc[..., 1:2]
        acc_leye = acc[..., 2:3]
        acc_reye = acc[..., 3:4]
        
        ########
        # compute loss
        loss_dict = {}
        loss = 0.

        # color and mask loss
        mask_scene = mask - leye_mask - reye_mask if ignore_eyeball else mask        
        loss_mask = F.l1_loss(acc_scene, mask_scene)
        loss_l1_recon = F.l1_loss(recon, pixels)
        loss_l1_pred = F.l1_loss(pred, pixels)        
        loss_dict.update({
            "loss_l1_recon": loss_l1_recon,
            "loss_l1_pred": loss_l1_pred,
            "loss_mask": loss_mask,
        })
        loss += (loss_l1_pred + loss_l1_recon) * cfg["loss"]["w_l1"] + loss_mask * cfg["loss"]["w_mask"]
        
        # loss_s3im = self.s3im_loss(recon, pixels)
        # loss += loss_s3im * cfg["loss"]["w_l1"]
        # loss_dict["loss_s3im"] = loss_s3im

        if sample_patch:  # compute LPIPS loss
            patch_size = 64
            num_patch = recon.shape[0] // (patch_size ** 2)
            recon_patch = recon.reshape(num_patch, patch_size, patch_size, 3).permute(0, 3, 1, 2)
            pixels_patch = pixels.reshape(num_patch, patch_size, patch_size, 3).permute(0, 3, 1, 2)
            loss_patch_lpips = trainer.lpips_loss(recon_patch, pixels_patch, normalize=True).mean()
            loss_dict.update({
                "loss_vol_lpips": loss_patch_lpips,
            })
            loss += loss_patch_lpips * cfg["loss"]["w_lpips"]

        # eikonal loss
        sdf_grad_samples = sample_attr_dict["sdf_grad"]
        eikonal = (torch.linalg.norm(sdf_grad_samples, ord=2, dim=-1) - 1.) ** 2
        loss_eikonal = eikonal.mean()
        loss_dict.update({
            "loss_eikonal": loss_eikonal,
            "beta": self.beta(),
            "geometry_hash_level": torch.tensor([self.geometry.level]),
            "num_rays": torch.tensor([num_rays]),
        })
        loss += loss_eikonal * cfg["loss"]["w_eikonal"]
        
        # regularize the composition of eyeball and face
        # sdf mask loss (from ObjectSDF++)
        if not ignore_eyeball:
            face_mask = mask - leye_mask - reye_mask
            loss_sdf_mask = F.l1_loss(acc_face, face_mask) + F.l1_loss(acc_leye, leye_mask) + \
                F.l1_loss(acc_reye, reye_mask)
            loss += loss_sdf_mask * cfg["loss"]["w_sdf_mask"]
            loss_dict["loss_sdf_mask"] = loss_sdf_mask

        # regularize the diffuse regions
        spec = attr_dict["specular"]
        loss_hair_diff = torch.mean(spec * data["diff_mask"])
        loss_dict["loss_hair_diff"] = loss_hair_diff
        loss += loss_hair_diff * cfg["loss"]["w_reg_diff"]

        # regularize the specular albedo to be similar with AlbedoMM fitting results
        albedomm_mask = mask_scene * (1 - data["diff_mask"])
        gt_spec_albedo = data["bfm_albedo"]
        loss_abdmm = F.l1_loss(spec * albedomm_mask * self.light_scale, gt_spec_albedo * albedomm_mask)
        loss_dict["loss_abdmm"] = loss_abdmm
        loss += loss_abdmm * cfg["loss"]["w_abdmm"]

        # regularize the smooth regions
        num_sample = len(sample_weights) / num_rays
        eps_weight = (
            data["smooth_mask"] * cfg["loss"]["w_eps_n_smooth"] + \
            mask_scene * (1 - data["smooth_mask"]) * cfg["loss"]["w_eps_n"]
        )[ray_idx.long()]
        n_eps = sample_attr_dict["eps_normal"]
        n = sample_attr_dict["normal"]
        loss_eps_n = torch.mean(eps_weight * (1.0 - torch.sum(n * n_eps, dim=-1, keepdim=True))) * num_sample
        loss_dict["loss_eps_n"] = loss_eps_n
        loss += loss_eps_n

        # regularize the specular smooth
        # num_sample = len(sample_weights) / num_rays
        # sa_eps = sample_attr_dict["eps_specular"]
        # sa = sample_attr_dict["specular"]
        # loss_eps_sa = torch.mean((sa - sa_eps).abs()) * num_sample
        # loss_dict["loss_eps_sa"] = loss_eps_sa
        # loss += loss_eps_sa * cfg["loss"]["w_eps_sa"]

        # update num of rays
        num_rays_update = int(
            num_rays
            * (cfg["renderer"]["sample_batch_size"] / max(float(n_rendering_samples), 16))
        )
        num_rays_update = (num_rays_update // 64) * 64
        num_rays_update = min(num_rays_update, cfg["renderer"].get("max_num_ray", 65536))
        trainer.train_dataset.update_num_rays(num_rays_update)
        
        loss_dict["loss"] = loss
        return loss, loss_dict

    def valid_step(self, trainer, inv_render=False, **kwargs):
        # parse params
        cfg = trainer.cfg
        data = trainer.val_data
        self.batch = data
        renderer = trainer.renderer
        render_step_size = trainer.render_step_size
        
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]
        
        ########
        # network forward
        attr, acc, depth, n_rendering_samples, sample_weights, sample_attrs, ray_idx = renderer.render_image(
            rays,
            # rendering options
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            test_chunk_size=cfg["renderer"]["chunk_size"],
            c2w=data["c2w"],
            **kwargs,
        )

        attr_dict = self.get_attr_dict(attr)
        diff = attr_dict["diff_shading"]
        spec = attr_dict["spec_shading"]
        normal = attr_dict["normal"]
        color = attr_dict["color"]

        # compute image loss
        mask = acc[..., :1].permute(2, 0, 1)[None, ...]
        mask_face = acc[..., 1:2].permute(2, 0, 1)
        mask_leye = acc[..., 2:3].permute(2, 0, 1)
        mask_reye = acc[..., 3:4].permute(2, 0, 1)
        color = color.permute(2, 0, 1)[None, ...]
        diff = diff.permute(2, 0, 1)[None, ...]
        spec = spec.permute(2, 0, 1)[None, ...]
        recon = (spec + diff) ** (1 / self.gamma)
        color = color ** (1 / self.gamma)
        color = recon if inv_render else color
        
        # vis
        diff_ldr = diff ** (1 / self.gamma)
        spec_ldr = spec
        gt = pixels.permute(2, 0, 1)[None, ...]
        normal = F.normalize(normal, dim=-1)
        normal = normal.permute(2, 0, 1)[None, ...]
        normal = (normal + 1) / 2
        return {
            "SSIM": trainer.ssim_loss(color, gt, window_size=3).mean().item(),
            "PSNR": trainer.psnr_loss(color, gt, max_val=1.).item(),
            "LPIPS": trainer.lpips_loss(color, gt, normalize=True).mean().item(),
            # "SSIM": trainer.ssim_loss(color ** (self.gamma), gt ** (self.gamma), window_size=3).mean().item(),
            # "PSNR": trainer.psnr_loss(color ** (self.gamma), gt ** (self.gamma), max_val=1.).item(),
            # "LPIPS": trainer.lpips_loss(color ** (self.gamma), gt ** (self.gamma), normalize=True).mean().item(),
            "vis": torch.cat([
                diff_ldr, spec_ldr * 3, recon, gt, normal, mask_leye * gt, mask_reye * gt, mask_face * gt
            ], dim=-1),
            "mask": mask,
        }

    def _mesh_step(self, vertices, faces, data, trainer):
        cam_int = torch.clone(data["intrinsic"])  # [3,3]
        cam_int[0] /= trainer.WIDTH
        cam_int[1] /= trainer.HEIGHT
        cam_int = cam_int[None, ...]  # [1,3,3]
        height, width = trainer.HEIGHT, trainer.WIDTH
        mesh_renderer = trainer.mesh_renderer

        train_data = trainer.train_data
        cfg = trainer.cfg

        cam_ext = torch.inverse(train_data["attr"]["c2w"][0])[:-1][None, ...]  # [1,3,4]
        cam_pos = train_data["origins"][0]  # [3]
        dist2 = torch.sum((vertices - cam_pos) ** 2, dim=-1, keepdim=True)
        mask = torch.ones_like(vertices[..., :1])
        attrs = torch.cat([vertices, mask, dist2], dim=-1)  # [1,v,c]
        
        mesh_dict = {
            "faces": faces,
            "vertice": vertices,
            "attributes": attrs,
            "size": (height, width),
        }
        
        with torch.no_grad():
            attr_img, pix_to_face = mesh_renderer.render_mesh(mesh_dict, cam_int, cam_ext)  # [1,2,h,w] [1,h,w,1]
        
        # compute img space attrs
        coord_img = attr_img[:, :3]  # [1,3,h,w]
        mask_img = attr_img[:, 3:4]  # [1,1,h,w]
        dist2_img = attr_img[:, 4:5].clip(min=1e-6)  # [1,1,h,w]
        coord_flat = coord_img.permute(0, 2, 3, 1).reshape(-1, 3)  # [hw,3]
        vis_mask = (mask_img.permute(0, 2, 3, 1) > 0).reshape(-1)  # [hw]
        coord_flat_vis = coord_flat[vis_mask]  # [nvis,3]

        # compute BRDF params
        x = coord_flat_vis
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        with torch.enable_grad():
            x.requires_grad_(True)
            sdf, feature = self.forward_geometry(x)
            leye_sdf, reye_sdf = self.eyeball(x)
            real_sdf = torch.minimum(reye_sdf, leye_sdf)
            real_sdf = torch.minimum(real_sdf, sdf)
            grad = torch.autograd.grad(
                real_sdf,
                x,
                grad_outputs=torch.ones_like(sdf),
                create_graph=False, 
                retain_graph=False, 
            )[0]
        diff, spec = self.forward_material(x.detach())
        with torch.no_grad():
            normal_flat_vis = F.normalize(grad, dim=-1)

        # shading
        diff_shading, spec_shading = self.shader.shading(
            pos_world=coord_flat_vis, diff_color=diff, spec_params=spec, normal=normal_flat_vis,
            view_pos_world=cam_pos, light_pos_world=None, c2w=train_data["c2w"],
        )
        diff_shading = diff_shading * self.light_intensity.to(x.device)  # [nvis,3]
        spec_shading = spec_shading * self.light_intensity.to(x.device)  # [nvis,3]
        diff_res = torch.zeros_like(coord_flat)
        spec_res = torch.zeros_like(coord_flat)
        spec_brdf_res = torch.zeros_like(coord_flat[..., :1])
        diff_res[vis_mask] = diff_shading
        spec_res[vis_mask] = spec_shading
        spec_brdf_res[vis_mask] = spec["specular"]
        diff_res = diff_res.reshape(1, height, width, -1).permute(0, 3, 1, 2) / dist2_img
        spec_res = spec_res.reshape(1, height, width, -1).permute(0, 3, 1, 2) / dist2_img
        spec_brdf_res = spec_brdf_res.reshape(1, height, width, -1).permute(0, 3, 1, 2)

        # compute loss
        pred = diff_res + spec_res  # [1,3,h,w]
        pixels = train_data["pixels"].reshape(1, height, width, -1).permute(0, 3, 1, 2) ** self.gamma
        diff_mask = train_data["diff_mask"].reshape(1, height, width, -1).permute(0, 3, 1, 2)
        gt_spec_albedo = data["bfm_albedo"].reshape(1, height, width, -1).permute(0, 3, 1, 2)
        
        loss_color_l1 = F.l1_loss(pred, pixels)
        loss_lpips = trainer.lpips_loss(pred, pixels, normalize=True).mean()
        loss_hair_diff = torch.sum(diff_mask * spec_brdf_res) / torch.sum(diff_mask).clamp(min=1e-3)
        albedomm_mask = mask_img * (1 - diff_mask)
        loss_abdmm = F.l1_loss(spec_brdf_res * albedomm_mask * self.light_scale, gt_spec_albedo * albedomm_mask)
        loss = 0.
        loss_dict = {}
        loss += loss_lpips * cfg["loss"]["w_lpips"] + \
            loss_color_l1 * cfg["loss"]["w_l1"] + \
            loss_hair_diff * cfg["loss"]["w_reg_diff"] + \
            loss_abdmm * cfg["loss"]["w_abdmm_refine"]
        loss_dict.update({
            "loss_mesh_l1": loss_color_l1,
            "loss_mesh_lpips": loss_lpips,
            "loss_mesh_hair_diff": loss_hair_diff,
            "loss_abdmm_refine": loss_abdmm,
        })
        return loss, loss_dict

    def train_mesh_step_precompute(self, trainer):
        data = trainer.train_data
        self.batch = data

        data = trainer.train_data
        vertices = trainer.precompute_vertices
        faces = trainer.precompute_faces
        return self._mesh_step(vertices, faces, data, trainer)

    def valid_mesh_step(self, trainer, **kwargs):
        data = trainer.val_data
        self.batch = data
        pixels = data["pixels"]
        gt_mask = data["mask"]
        
        vertices = trainer.precompute_vertices
        faces = trainer.precompute_faces

        msaa = 2
        cam_int = torch.clone(data["intrinsic"])  # [3,3]
        cam_int[0] /= trainer.WIDTH
        cam_int[1] /= trainer.HEIGHT
        cam_int = cam_int[None, ...]  # [1,3,3]
        height, width = trainer.HEIGHT * msaa, trainer.WIDTH * msaa
        mesh_renderer = trainer.mesh_renderer

        val_data = trainer.val_data
        cfg = trainer.cfg

        cam_ext = torch.inverse(val_data["attr"]["c2w"][0])[:-1][None, ...]  # [1,3,4]
        cam_pos = val_data["origins"][0, 0]  # [3]
        dist2 = torch.sum((vertices - cam_pos) ** 2, dim=-1, keepdim=True)
        mask = torch.ones_like(vertices[..., :1])
        attrs = torch.cat([vertices, mask, dist2], dim=-1)  # [1,v,c]
        
        mesh_dict = {
            "faces": faces,
            "vertice": vertices,
            "attributes": attrs,
            "size": (height, width),
        }
        
        with torch.no_grad():
            attr_img, pix_to_face = mesh_renderer.render_mesh(mesh_dict, cam_int, cam_ext)  # [1,2,h,w] [1,h,w,1]
        
        # compute img space attrs
        coord_img = attr_img[:, :3]  # [1,3,h,w]
        mask_img = attr_img[:, 3:4]  # [1,1,h,w]
        dist2_img = attr_img[:, 4:5].clip(min=1e-6)  # [1,1,h,w]
        coord_flat = coord_img.permute(0, 2, 3, 1).reshape(-1, 3)  # [hw,3]
        vis_mask = (mask_img.permute(0, 2, 3, 1) > 0).reshape(-1)  # [hw]
        coord_flat_vis = coord_flat[vis_mask]  # [nvis,3]

        # compute BRDF params
        x = coord_flat_vis
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        with torch.enable_grad():
            x.requires_grad_(True)
            sdf, feature = self.forward_geometry(x)
            leye_sdf, reye_sdf = self.eyeball(x)
            real_sdf = torch.minimum(reye_sdf, leye_sdf)
            real_sdf = torch.minimum(real_sdf, sdf)
            eye_mask = (real_sdf < sdf).float()
            grad = torch.autograd.grad(
                real_sdf,
                x,
                grad_outputs=torch.ones_like(sdf),
                create_graph=False, 
                retain_graph=False, 
            )[0]
        diff, spec = self.forward_material(x.detach())
        spec["specular"] = spec["specular"] * (1 - eye_mask) + eye_mask * 0.2
        spec["roughness"] = spec["roughness"] * (1 - eye_mask) + eye_mask * 0.05
        with torch.no_grad():
            normal_flat_vis = F.normalize(grad, dim=-1)

        # shading
        diff_shading, spec_shading = self.shader.shading(
            pos_world=coord_flat_vis, diff_color=diff, spec_params=spec, normal=normal_flat_vis,
            view_pos_world=cam_pos, light_pos_world=None, c2w=val_data["c2w"],
        )
        diff_shading = diff_shading * self.light_intensity.to(x.device)  # [nvis,3]
        spec_shading = spec_shading * self.light_intensity.to(x.device)  # [nvis,3]
        diff_res = torch.zeros_like(coord_flat)
        spec_res = torch.zeros_like(coord_flat)
        normal_res = torch.zeros_like(coord_flat)
        spec_brdf_res = torch.zeros_like(coord_flat[..., :1])
        diff_res[vis_mask] = diff_shading
        spec_res[vis_mask] = spec_shading
        normal_res[vis_mask] = normal_flat_vis
        spec_brdf_res[vis_mask] = spec["specular"]
        diff_res = diff_res.reshape(1, height, width, -1).permute(0, 3, 1, 2) / dist2_img
        spec_res = spec_res.reshape(1, height, width, -1).permute(0, 3, 1, 2) / dist2_img
        normal_res = normal_res.reshape(1, height, width, -1).permute(0, 3, 1, 2)
        spec_brdf_res = spec_brdf_res.reshape(1, height, width, -1).permute(0, 3, 1, 2)

        # compute image loss
        spec_res = F.avg_pool2d(spec_res, kernel_size=msaa, stride=msaa)
        diff_res = F.avg_pool2d(diff_res, kernel_size=msaa, stride=msaa)
        normal_res = F.avg_pool2d(normal_res, kernel_size=msaa, stride=msaa)
        mask_img = F.avg_pool2d(mask_img, kernel_size=msaa, stride=msaa)
        color = (spec_res + diff_res) ** (1 / self.gamma)
        
        # vis
        diff_ldr = diff_res ** (1 / self.gamma)
        spec_ldr = spec_res.clamp(min=0.01) ** (1 / self.gamma)
        gt = pixels.permute(2, 0, 1)[None, ...]
        gt_mask = gt_mask.permute(2, 0, 1)[None, ...]
        normal = F.normalize(normal_res, dim=1)
        normal = (normal + 1) / 2
        return {
            "SSIM": trainer.ssim_loss(color, gt, window_size=3).mean().item(),
            "PSNR": trainer.psnr_loss(color, gt, max_val=1.).item(),
            "LPIPS": trainer.lpips_loss(color, gt, normalize=True).mean().item(),
            # "SSIM": trainer.ssim_loss(color ** (self.gamma), gt ** (self.gamma), window_size=3).mean().item(),
            # "PSNR": trainer.psnr_loss(color ** (self.gamma), gt ** (self.gamma), max_val=1.).item(),
            # "LPIPS": trainer.lpips_loss(color ** (self.gamma), gt ** (self.gamma), normalize=True).mean().item(),
            "vis": torch.cat([
                diff_ldr, spec_ldr * 3, color, gt, normal,
            ], dim=-1),
            "mask": mask_img,
            "diff_ldr": diff_ldr,
            "spec_ldr": spec_ldr,
            "recon": color,
            "gt": gt,
            "gt_mask": gt_mask,
            "normal": normal,
        }

    def compute_mask_step(self, trainer, ignore_eyeball=False):
        '''
        只在clean mesh时被调用
        '''
        # parse params
        cfg = trainer.cfg
        data = trainer.val_data
        self.batch = data
        renderer = trainer.renderer
        render_step_size = trainer.render_step_size
        
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        
        ########
        # network forward
        attr, acc, depth, n_rendering_samples, sample_weights, sample_attrs, ray_idx = renderer.render_image(
            rays,
            # rendering options
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            # test options
            test_chunk_size=cfg["renderer"]["chunk_size"],
            # other options
            c2w=data["c2w"],
            only_render_mask=True,
            ignore_eyeball=ignore_eyeball,
        )
        # compute image loss
        mask = acc[..., :1].permute(2, 0, 1)[None, ...]
        return {
            "mask": mask,
        }


def sdf_to_density(sdf, beta):
    x = -sdf

    # select points whose x is smaller than 0: 1 / beta * 0.5 * exp(x/beta)
    ind0 = x <= 0
    val0 = 1 / beta * (0.5 * torch.exp(x[ind0] / beta))

    # select points whose x is bigger than 0: 1 / beta * (1 - 0.5 * exp(-x/beta))
    ind1 = x > 0
    val1 = 1 / beta * (1 - 0.5 * torch.exp(-x[ind1] / beta))

    val = torch.zeros_like(sdf)
    val[ind0] = val0
    val[ind1] = val1

    return val
