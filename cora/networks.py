import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import tinycudann as tcnn

from .disney_brdf import _apply_shading_burley


class Shader(nn.Module):
    def __init__(
        self,
        brdf_cfg,
    ):
        super().__init__()
        self.brdf_type = brdf_cfg["type"]

    def shading(self, diff_color, spec_params, normal, pos_world, view_pos_world, light_pos_world, c2w):
        '''
        diff_color: diffuse albedo
        spec_params: specular BRDF coefficients
        pos: position of sample point
        view_pos: position of camera
        light_pos: position of light source
        '''
        # render under point light source        
        if self.brdf_type == "blender":
            return self._blender_shading(diff_color, spec_params, normal, pos_world, view_pos_world, light_pos_world, c2w)
        else:
            raise NotImplementedError
    
    def _blender_shading(self, diff_color, spec_params, normal, pos_world, view_pos_world, light_pos_world, c2w):
        view_dir = view_pos_world - pos_world  # from surface point to camera
        view_dir = F.normalize(view_dir, dim=-1)
        
        if light_pos_world is None:
            light_pos_world = view_pos_world
            # light_pos_cam = torch.zeros_like(view_pos_world)
            # light_pos_world = (c2w[:3, :3] @ light_pos_cam[..., None] + c2w[:3, 3:])[..., 0]  # [3] or [n,3]
        
        light_dir = light_pos_world - pos_world  # from surface point to light source
        light_dir = F.normalize(light_dir, dim=-1)
        
        diff_shading, spec_shading = _apply_shading_burley(
            normals=normal,
            view_dirs=-view_dir,  # from camera to surface
            light_dirs=-light_dir,  # from light to surface
            specular=spec_params["specular"],
            base_color=diff_color,
            roughness=spec_params["roughness"],
        )
        return diff_shading, spec_shading.expand_as(diff_shading)

    def forward(self):
        pass


class VolumeMaterial(nn.Module):
    '''
    map a 3d position to its BRDF parameters
    '''
    def __init__(
        self,
        num_levels,
        level_dim,
        per_level_scale,
        brdf_cfg,
    ):
        super().__init__()
        self.encoding = tcnn.Encoding(
            n_input_dims=3, 
            encoding_config={
                'otype': 'HashGrid', 
                'n_levels': num_levels, 
                'n_features_per_level': level_dim, 
                'log2_hashmap_size': 14, 
                'base_resolution': 16, 
                'per_level_scale': per_level_scale, 
            }
        )
        
        self.brdf_type = brdf_cfg["type"]
        if self.brdf_type == "blender":
            self.len_diff, self.len_spec = 3, 2
            self.min_roughness = brdf_cfg["min_roughness"]
            self.max_roughness = brdf_cfg["max_roughness"]
            self.max_specular = brdf_cfg.get("max_specular", 1.)
            self.min_specular = brdf_cfg.get("min_specular", 0.)
        else:
            raise NotImplementedError
        
        n_output_dims = self.len_diff + self.len_spec
        self.network = tcnn.Network(
            n_input_dims=self.encoding.n_output_dims + 3,
            n_output_dims=n_output_dims,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )
    
    def forward(self, x):
        h = self.encoding(x).to(x)
        h = torch.cat([2 * x - 1, h], dim=-1)
        out = self.network(h).to(h)
        diff, spec = out[..., :self.len_diff], out[..., self.len_diff:]
        
        if self.brdf_type == "blender":
            diff = torch.sigmoid(diff)
            spec = torch.sigmoid(spec)
            roughness = spec[..., 1:2]
            specular = spec[..., :1]
            specular = specular * self.min_specular + (1 - specular) * self.max_specular
            roughness = roughness * self.min_roughness + (1 - roughness) * self.max_roughness
            spec = {
                "specular": specular,
                "roughness": roughness,
            }

        return diff, spec


class EyeballSDF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bound = cfg["bound"]
        self.init_eyeball(cfg)
    
    def init_eyeball(self, cfg):
        left_init = torch.tensor(cfg.get("eyeball_left_center", [0., 0., 0.])).cuda()
        right_init = torch.tensor(cfg.get("eyeball_right_center", [0., 0., 0.])).cuda()
        left_init = (left_init + 1) / 2
        right_init = (right_init + 1) / 2
        self.left_eyeball_center = torch.nn.parameter.Parameter(
            left_init, requires_grad=not cfg.get("fix_eyeball_center", False)
        )  # in hashgrid space [0,1]^3
        self.right_eyeball_center = torch.nn.parameter.Parameter(
            right_init, requires_grad=not cfg.get("fix_eyeball_center", False)
        )
        self.eyeball_size = torch.nn.parameter.Parameter(
            torch.zeros(1).cuda(), requires_grad=not cfg.get("fix_eyeball_center", False)
        )
        self.eyeball_min_size = cfg["eyeball_min_size"]  # in world space [-bound,bound]^3
        self.eyeball_max_size = cfg["eyeball_max_size"]

    def get_world_eyeball_center(self):
        return (
            (2 * self.left_eyeball_center - 1) * self.bound,
            (2 * self.right_eyeball_center - 1) * self.bound,
        )

    def get_world_eyeball_size(self):
        '''
        the whole model in [-bound,bound]^3
        '''
        low_size, high_size = self.eyeball_min_size, self.eyeball_max_size
        eyeball_size = torch.sigmoid(self.eyeball_size) * (high_size - low_size) + low_size
        return eyeball_size

    def get_normal(self, x):
        return (
            F.normalize(x - self.left_eyeball_center, dim=-1),
            F.normalize(x - self.right_eyeball_center, dim=-1),
        )

    def get_eyeball_size(self):
        '''
        the whole model in [0,1]^3
        '''
        scale = 1 / (2 * self.bound)
        low_size, high_size = scale * self.eyeball_min_size, scale * self.eyeball_max_size
        eyeball_size = torch.sigmoid(self.eyeball_size) * (high_size - low_size) + low_size
        return eyeball_size

    def forward(self, x):
        eyeball_size = self.get_eyeball_size()
        reye_sdf = torch.sum((x - self.right_eyeball_center) ** 2, dim=-1, keepdim=True).sqrt() - eyeball_size
        leye_sdf = torch.sum((x - self.left_eyeball_center) ** 2, dim=-1, keepdim=True).sqrt() - eyeball_size
        return leye_sdf, reye_sdf


class VolumeRadianceHead(nn.Module):
    def __init__(
        self,
        input_feature_dim=16,
    ):
        super().__init__()
        self.n_output_dims = 3
        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                'otype': 'SphericalHarmonics', 
                'degree': 4
            },
        )
        self.n_input_dims = input_feature_dim + self.encoding.n_output_dims + 3 + 3
        self.network = tcnn.Network(
            n_input_dims=self.n_input_dims,
            n_output_dims=self.n_output_dims, 
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'none',
                'n_neurons': 64,
                'n_hidden_layers': 2,
            },
        )
    
    def forward(self, features, dirs, normal):
        dirs = (dirs + 1.) / 2.
        dirs_embd = self.encoding(dirs).to(dirs)
        dirs_embd = torch.cat([dirs * 2 - 1, dirs_embd], dim=-1)
        network_inp = torch.cat([features, dirs_embd, normal], dim=-1)
        color = self.network(network_inp).to(dirs)
        color = torch.sigmoid(color)
        return color


class VanillaMLP(nn.Module):
    def __init__(
        self, 
        dim_in, 
        dim_out, 
        sphere_init=True,
        sphere_init_radius=0.5,
        weight_norm=True,
        n_neurons=64,
        n_hidden_layers=1
    ):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = n_neurons, n_hidden_layers
        self.sphere_init, self.weight_norm = sphere_init, weight_norm
        self.sphere_init_radius = sphere_init_radius
        self.layers = [
            self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False), self.make_activation()
        ]
        for i in range(self.n_hidden_layers - 1):
            self.layers += [
                self.make_linear(self.n_neurons, self.n_neurons, is_first=False, is_last=False), self.make_activation()
            ]
        self.layers += [self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)]
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x
    
    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=True) # network without bias will degrade quality
        if self.sphere_init:
            if is_last:
                torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                torch.nn.init.normal_(layer.weight, mean=math.sqrt(math.pi) / math.sqrt(dim_in), std=0.0001)
            elif is_first:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                torch.nn.init.normal_(layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(dim_out))
            else:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(dim_out))
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        
        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer   

    def make_activation(self):
        if self.sphere_init:
            return nn.Softplus(beta=100)
        else:
            return nn.ReLU(inplace=True)


class VolumeSDF(nn.Module):
    def __init__(
        self,
        num_levels,
        level_dim,
        per_level_scale,
        n_output_dims=13,
    ):
        super().__init__()
        self.n_output_dims = n_output_dims
        self.encoding = tcnn.Encoding(
            n_input_dims=3, 
            encoding_config={
                'otype': 'HashGrid', 
                'n_levels': num_levels, 
                'n_features_per_level': level_dim, 
                'log2_hashmap_size': 14, 
                'base_resolution': 16, 
                'per_level_scale': per_level_scale, 
                "interpolation": "Smoothstep",
            }
        )
        self.network = VanillaMLP(
            dim_in=self.encoding.n_output_dims + 3,
            dim_out=self.n_output_dims,
            n_hidden_layers=2,
        )
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.level_mask = torch.ones(self.num_levels * self.level_dim)

    def update_level(self, level):
        '''
        generate mask (self.level_mask) to deactivate the hash-grid feature
        '''
        self.level = min(level, self.num_levels)
        ones = torch.ones(self.level, self.level_dim)
        if self.num_levels > level:
            zeros = torch.zeros(self.num_levels - level, self.level_dim)
            self.level_mask = torch.cat([ones, zeros], dim=0).reshape(-1)
        else:
            self.level_mask = ones.reshape(-1)
    
    def forward(self, points):
        h = self.encoding(points).to(points) * self.level_mask.to(points.device)
        h = torch.cat([2 * points - 1, h], dim=-1)
        out = self.network(h).to(h)
        sdf, feature = out[..., :1], out
        return sdf, feature


class BetaNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta_min = 0.0001
        init_val = 0.1
        self.register_parameter('beta', nn.Parameter(torch.tensor(init_val)))

    def forward(self):
        beta = self.beta.abs() + self.beta_min
        return beta
