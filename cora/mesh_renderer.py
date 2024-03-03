import torch
import torch.nn.functional as F
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.structures import Meshes
from pytorch3d.ops import interpolate_face_attributes


class MeshRenderer:
    def __init__(self, device):
        self.device = device
    
    def render_ndc(self, mesh_dict):
        vertice_ndc = mesh_dict["vertice"]  # [b,h*w,3]
        faces = mesh_dict["faces"]  # [b,nface,3]
        attributes = mesh_dict["attributes"]  # [b,h*w,4]
        h, w = mesh_dict["size"]

        ############
        # render
        mesh = Meshes(vertice_ndc, faces)
        pix_to_face, _, bary_coords, _ = rasterize_meshes(mesh, (h, w), faces_per_pixel=1, blur_radius=0)  # [b,h,w,1] [b,h,w,1,3]

        # mask = pix_to_face > -1
        # save_image(mask.float().permute(0, 3, 1, 2), "pix_to_face.png")

        b, nf, _ = faces.size()
        c = attributes.shape[-1]
        faces = faces.reshape(b, nf * 3, 1).repeat(1, 1, c)  # [b,3f,4]
        face_attributes = torch.gather(attributes, dim=1, index=faces)  # [b,3f,4]
        face_attributes = face_attributes.reshape(b * nf, 3, c)  # in pytorch3d, the index of mesh#2's first vertex is set to nface
        output = interpolate_face_attributes(pix_to_face, bary_coords, face_attributes)
        output = output.squeeze(-2).permute(0, 3, 1, 2)
        return output, pix_to_face
    
    def render_mesh(self, mesh_dict, cam_int, cam_ext):
        '''
        input:
            mesh: the output for construct_mesh function
            cam_int: [b,3,3] normalized camera intrinsic matrix
            cam_ext: [b,3,4] camera extrinsic matrix with the same scale as depth map

            camera coord: x to right, z to front, y to down
        
        output:
            render: [b,3,h,w]
        '''
        vertice = mesh_dict["vertice"]  # [b,h*w,3]
        faces = mesh_dict["faces"]  # [b,nface,3]
        attributes = mesh_dict["attributes"]  # [b,h*w,4]
        h, w = mesh_dict["size"]

        ############
        # to NDC space
        vertice_homo = self.lift_to_homo(vertice)  # [b,h*w,4]
        # [b,1,3,4] x [b,h*w,4,1] = [b,h*w,3,1]
        vertice_world = torch.matmul(cam_ext.unsqueeze(1), vertice_homo[..., None]).squeeze(-1)  # [b,h*w,3]
        self.near_z = torch.min(vertice_world[..., -1]).item() / 2
        self.far_z = torch.max(vertice_world[..., -1]).item() * 2
        # [b,1,3,3] x [b,h*w,3,1] = [b,h*w,3,1]        
        vertice_world_homo = self.lift_to_homo(vertice_world)
        persp = self.get_perspective_from_intrinsic(cam_int)  # [b,4,4]

        # [b,1,4,4] x [b,h*w,4,1] = [b,h*w,4,1]
        vertice_ndc = torch.matmul(persp.unsqueeze(1), vertice_world_homo[..., None]).squeeze(-1)  # [b,h*w,4]
        vertice_ndc = vertice_ndc[..., :-1] / vertice_ndc[..., -1:]
        vertice_ndc[..., :-1] *= -1
        vertice_ndc[..., 1] *= h / w

        ############
        # render
        mesh = Meshes(vertice_ndc, faces)
        pix_to_face, _, bary_coords, _ = rasterize_meshes(mesh, (h, w), faces_per_pixel=1, blur_radius=0)  # [b,h,w,1] [b,h,w,1,3]

        # mask = pix_to_face > -1
        # save_image(mask.float().permute(0, 3, 1, 2), "pix_to_face.png")

        b, nf, _ = faces.size()
        c = attributes.shape[-1]
        faces = faces.reshape(b, nf * 3, 1).repeat(1, 1, c)  # [b,3f,4]
        face_attributes = torch.gather(attributes, dim=1, index=faces)  # [b,3f,4]
        face_attributes = face_attributes.reshape(b * nf, 3, c)  # in pytorch3d, the index of mesh#2's first vertex is set to nface
        output = interpolate_face_attributes(pix_to_face, bary_coords, face_attributes)
        output = output.squeeze(-2).permute(0, 3, 1, 2)
        return output, pix_to_face
    
    def lift_to_homo(self, coord):
        '''
        return the homo version of coord

        input: coord [..., k]
        output: homo_coord [...,k+1]
        '''
        ones = torch.ones_like(coord[..., -1:])
        return torch.cat([coord, ones], dim=-1)

    def get_perspective_from_intrinsic(self, cam_int):
        '''
        input:
            cam_int: [b,3,3]
        
        output:
            persp: [b,4,4]
        '''
        fx, fy = cam_int[:, 0, 0], cam_int[:, 1, 1]  # [b]
        cx, cy = cam_int[:, 0, 2], cam_int[:, 1, 2]  # [b]

        one = torch.ones_like(cx)  # [b]
        zero = torch.zeros_like(cx)  # [b]

        near_z, far_z = self.near_z * one, self.far_z * one
        a = (near_z + far_z) / (far_z - near_z)
        b = -2.0 * near_z * far_z / (far_z - near_z)

        matrix = [[2.0 * fx, zero, 2.0 * cx - 1.0, zero],
                  [zero, 2.0 * fy, 2.0 * cy - 1.0, zero],
                  [zero, zero, a, b],
                  [zero, zero, one, zero]]
        # -> [[b,4],[b,4],[b,4],[b,4]] -> [b,4,4]        
        persp = torch.stack([torch.stack(row, dim=-1) for row in matrix], dim=-2)  # [b,4,4]
        return persp
