'''
Disney BRDF implementation

borrowed from https://github.com/za-cheng/WildLight
'''


import torch
import torch.nn.functional as F
from numpy import pi


def dot(tensor1, tensor2, dim=-1, keepdim=False, non_negative=False, epsilon=1e-6) -> torch.Tensor:
    x =  (tensor1 * tensor2).sum(dim=dim, keepdim=keepdim)
    if non_negative:
        x = torch.clamp_min(x, epsilon)
    return x


def _GGX_smith(n_d_h, n_d_v, n_d_l, alpha, epsilon=1e-10):
    n_d_h_sq = n_d_h ** 2
    alpha_sq = alpha ** 2
    D = alpha_sq / pi / (n_d_h_sq * (alpha_sq - 1) + 1 + epsilon) ** 2 # GGX
    
    n_d_v_sq = n_d_v ** 2
    n_d_l_sq = n_d_l ** 2
    G_v = 2 / ( torch.sqrt(1 + alpha_sq * (1 / n_d_v_sq - 1)) + 1)
    G_l = 2 / ( torch.sqrt(1 + alpha_sq * (1 / n_d_l_sq - 1)) + 1)
    G = G_v * G_l
    return D, G


def _diffuse(n_d_l, n_d_v, l_d_h, roughness):
    F_D90 = 2 * roughness * (l_d_h ** 2) + 0.5
    diff_l = 1 + (F_D90 - 1) * ((1 - n_d_l) ** 5)
    diff_v = 1 + (F_D90 - 1) * ((1 - n_d_v) ** 5)
    base_diffuse = diff_l * diff_v / pi
    return base_diffuse


def _burley_shading(
    normal_vecs, 
    incident_vecs,  # surface to light
    view_vecs,  # surface to camera
    specular,
    roughness,
    base_color,
):

    half_vecs = torch.nn.functional.normalize(incident_vecs + view_vecs, dim=-1)
    h_n = dot(half_vecs, normal_vecs, non_negative=True, keepdim=True) # (..., 1)
    v_n = dot(view_vecs, normal_vecs, non_negative=True, keepdim=True) # (..., 1)
    l_n = dot(incident_vecs, normal_vecs, non_negative=True, keepdim=True) # (..., 1)
    l_h = dot(incident_vecs, half_vecs, non_negative=True, keepdim=True) # (..., 1)
    
    alpha = 0.0001 + (roughness ** 2) * (1 - 0.0002)

    D_metal, G_metal = _GGX_smith(h_n, v_n, l_n, alpha) #(..., 1)
    F0 = specular * 0.08 # (..., 3)
    F_metal = F0 + (1 - F0) * ((1 - l_h) ** 5)
    # F_metal = F0
    r_specular = D_metal * G_metal * F_metal / (4 * v_n * l_n) # (..., 3)

    r_diffuse = _diffuse(l_n, v_n, l_h, roughness) * base_color #(..., 3)
    return r_diffuse, r_specular


def _apply_shading_burley(
    normals,
    view_dirs,  # camera to surface
    light_dirs,  # light to surface
    specular,
    roughness,
    base_color,
):
    normals = F.normalize(normals, dim=-1)
    light_dirs_ = F.normalize(light_dirs, dim=-1)
    view_dirs = F.normalize(view_dirs, dim=-1)

    falloff = F.relu(-(normals * light_dirs_).sum(-1)) # (...)
    forward_facing = dot(normals, view_dirs) < 0
    visible_mask = ((falloff > 0) & forward_facing) # (...) boolean
    falloff = torch.where(visible_mask, falloff, torch.zeros(1, device=falloff.device)) # (...) cosine falloff, 0 if not visible

    diffuse, non_diffuse = _burley_shading(normals, -light_dirs_, -view_dirs, specular, roughness, base_color)
    
    return diffuse * falloff[..., None], non_diffuse * falloff[..., None]


if __name__ == "__main__":
    normals = torch.randn(10, 3)
    view_dirs = torch.randn(10, 3)
    light_dirs = torch.randn(10, 3)
    specular = torch.sigmoid(torch.randn(10, 1))
    roughness = torch.sigmoid(torch.randn(10, 1))
    base_color = torch.sigmoid(torch.randn(10, 3))
    diff, spec = _apply_shading_burley(
        normals, view_dirs, light_dirs, specular, roughness, base_color
    )
    print(diff.size(), spec.size())
