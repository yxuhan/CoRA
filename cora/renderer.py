from typing import Callable, Optional, Tuple
import collections
import torch
import torch.nn.functional as F
from nerfacc import (
    OccupancyGrid, 
    ray_marching, 
    accumulate_along_rays, 
)
from nerfacc.vol_rendering import render_transmittance_from_density
from nerfacc.pack import pack_info


Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))


def render_weight_from_density_obj_sdf(
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    sigmas_scene: torch.Tensor,
    sigmas_obj: torch.Tensor,
    *,
    packed_info: Optional[torch.Tensor] = None,
    ray_indices: Optional[torch.Tensor] = None,
    n_rays: Optional[int] = None,
) -> torch.Tensor:
    assert (
        ray_indices is not None or packed_info is not None
    ), "Either ray_indices or packed_info should be provided."
    if packed_info is None:
        packed_info = pack_info(ray_indices, n_rays=n_rays)
    
    transmittance = render_transmittance_from_density(
        t_starts, t_ends, sigmas_scene, ray_indices=ray_indices, n_rays=n_rays,
    )
    
    alphas_scene = 1.0 - torch.exp(-sigmas_scene * (t_ends - t_starts))
    weights_scene = transmittance * alphas_scene

    alphas_obj = 1.0 - torch.exp(-sigmas_obj * (t_ends - t_starts))
    weights_obj = transmittance * alphas_obj
    
    return weights_scene, weights_obj


class ObjSDFRenderer:
    def __init__(
        self,
        radiance_field: torch.nn.Module,
        occupancy_grid: OccupancyGrid,
        scene_aabb: torch.Tensor,
        **kwargs,
    ):
        super().__init__()
        self.radiance_field = radiance_field
        self.occupancy_grid = occupancy_grid
        self.scene_aabb = scene_aabb

    def render_image(
        self,
        rays: Rays,
        # rendering options
        near_plane: Optional[float] = None,
        far_plane: Optional[float] = None,
        render_step_size: float = 1e-3,
        render_bkgd: Optional[torch.Tensor] = None,
        alpha_thre: float = 0.0,
        # test options
        test_chunk_size: Optional[int] = None,
        **kwargs,
    ):
        """Render the pixels of an image."""
        rays_shape = rays.origins.shape
        if len(rays_shape) == 3:
            height, width, _ = rays_shape
            num_rays = height * width
            rays = namedtuple_map(
                lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
            )
        else:
            num_rays, _ = rays_shape

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = chunk_rays.origins[ray_indices]
            t_dirs = chunk_rays.viewdirs[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
            return self.radiance_field.query_density(positions, **kwargs)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = chunk_rays.origins[ray_indices]
            t_dirs = chunk_rays.viewdirs[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
            dist2 = torch.sum((positions - t_origins) ** 2, dim=-1, keepdim=True)
            return self.radiance_field(
                positions, view_dir=t_dirs, view_pos=t_origins, dist2=dist2, ray_idx=ray_indices, **kwargs
            )

        results = []
        chunk = torch.iinfo(torch.int32).max if test_chunk_size is None else test_chunk_size
        point_weights, point_attrs, ray_idx = [], [], []
        for i in range(0, num_rays, chunk):
            chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
            ray_indices, t_starts, t_ends = ray_marching(
                chunk_rays.origins,
                chunk_rays.viewdirs,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid,
                sigma_fn=sigma_fn,
                near_plane=near_plane,
                far_plane=far_plane,
                render_step_size=render_step_size,
                stratified=self.radiance_field.training,
                alpha_thre=alpha_thre,
            )
            attribute, opacity, depth, weights, attrs  = self.rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=chunk_rays.origins.shape[0],
                attr_sigma_fn=rgb_sigma_fn,
                render_bkgd=render_bkgd,
            )
            chunk_results = [attribute, opacity, depth, len(t_starts)]
            results.append(chunk_results)
            point_attrs.append(attrs)
            point_weights.append(weights)
            ray_idx.append(ray_indices)

        attributes, opacities, depths, n_rendering_samples = [
            torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
            for r in zip(*results)
        ]
        return (
            attributes.view((*rays_shape[:-1], -1)),
            opacities.view((*rays_shape[:-1], -1)),
            depths.view((*rays_shape[:-1], -1)),
            sum(n_rendering_samples),
            torch.cat(point_weights, dim=0),
            torch.cat(point_attrs, dim=0),
            torch.cat(ray_idx, dim=0),
        )

    def rendering(
        self,
        # ray marching results
        t_starts: torch.Tensor,
        t_ends: torch.Tensor,
        ray_indices: torch.Tensor,
        n_rays: int,
        # radiance field
        attr_sigma_fn: Callable,
        # rendering options
        render_bkgd: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Query sigma/alpha and other attribtues (e.g. color, normal, etc) with gradients
        attrs, sigmas_scene, sigmas_obj = attr_sigma_fn(t_starts, t_ends, ray_indices.long(), **kwargs)
        
        # Rendering: compute weights.
        weights, weights_obj = render_weight_from_density_obj_sdf(
            t_starts,
            t_ends,
            sigmas_scene=sigmas_scene,
            sigmas_obj=sigmas_obj,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )

        # Rendering: accumulate attrs, opacities, and depths along the rays.
        attributes = accumulate_along_rays(
            weights, ray_indices, values=attrs, n_rays=n_rays
        )
        opacities = accumulate_along_rays(
            weights, ray_indices, values=None, n_rays=n_rays
        )
        opacities_obj = accumulate_along_rays(
            weights_obj, ray_indices, values=None, n_rays=n_rays
        )
        depths = accumulate_along_rays(
            weights,
            ray_indices,
            values=(t_starts + t_ends) / 2.0,
            n_rays=n_rays,
        )
        opacities = torch.cat([opacities, opacities_obj], dim=-1)

        # Background composition.
        # if render_bkgd is not None:
        #     colors = colors + render_bkgd * (1.0 - opacities)

        return attributes, opacities, depths, weights, attrs
