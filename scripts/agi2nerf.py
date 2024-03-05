'''
this code is heavily borrowed from https://github.com/EnricoAhlers/agi2nerf
'''


import argparse
import xml.etree.ElementTree as ET
import math
import cv2
import numpy as np
import os
import json
import trimesh
from pytorch3d.io import load_obj


###############################################################################
# START
# code taken from https://github.com/NVlabs/instant-ngp
#Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

def central_point(out, args):
	# find a central point they are all looking at
	print("computing center of attention...")
	totw = 0.0
	totp = np.array([0.0, 0.0, 0.0])
	for f in out["frames"]:
		mf = np.array(f["transform_matrix"])[0:3,:]
		for g in out["frames"]:
			mg = g["transform_matrix"][0:3,:]
			p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
			if w > 0.0001:
				totp += p*w
				totw += w
	totp /= totw
	print(totp) # the cameras are looking at totp

	verts_torch, topo, _ = load_obj(args.mesh_path)
	verts = verts_torch.numpy()
	faces = topo.verts_idx.numpy()
	# mesh = trimesh.load_mesh(args.mesh_path)
	# verts = mesh.geometry["model"].vertices
	center = np.mean(verts, axis=0, keepdims=True)
	verts = verts - center
	scale = np.max(verts, axis=0) - np.min(verts, axis=0)
	s = np.max(scale) / 2 * 1.2
	for f in out["frames"]:
		f["transform_matrix"][0:3,3] -= totp
		f["transform_matrix"][0:3,3] /= s
		f["transform_matrix"] = f["transform_matrix"].tolist()
	
	cur_trans = np.array([totp[1], totp[2] , totp[0]])
	new_mesh = trimesh.Trimesh(
		vertices=verts_torch.numpy() / s - cur_trans / s,
		faces=faces,
	)
	new_mesh.export(os.path.join(args.save_mesh_path))
	return out

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	if image is None:
		print("Image not found:", imagePath)
		return 0
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = cv2.Laplacian(gray, cv2.CV_64F).var()
	return fm

#END
###############################################################################

#Copyright (C) 2022, Enrico Philip Ahlers. All rights reserved.

def parse_args():
	parser = argparse.ArgumentParser(description="convert Agisoft XML export to nerf format transforms.json")

	parser.add_argument("--data_root", help="specify xml file location")
	parser.add_argument("--out", default="transforms.json", help="output path")
	parser.add_argument("--imgtype", default="png", help="type of images (ex. jpg, png, ...)")
	args = parser.parse_args()
	args.imgfolder = os.path.join(args.data_root, "image")
	# args.xml_in = os.path.join(args.data_root, "output/project.files/0/doc.xml")
	args.mesh_path = os.path.join(args.data_root, "metashape_recon.obj")
	args.save_mesh_path = os.path.join(args.data_root, "metashape_recon_scaled.obj")
	return args



def get_calibration(root):
	for sensor in root[0][0]:
		for child in sensor:
			if child.tag == "calibration":
				return child
	print("No calibration found")	
	return None

def reflectZ():
	return [[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, -1, 0],
			[0, 0, 0, 1]]

def reflectY():
	return [[1, 0, 0, 0],
			[0, -1, 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1]]

def matrixMultiply(mat1, mat2):
	return np.array([[sum(a*b for a,b in zip(row, col)) for col in zip(*mat2)] for row in mat1])


if __name__ == "__main__":
	args = parse_args()
	XML_LOCATION = os.path.join(args.data_root, "metashape_recon.xml") 
	IMGTYPE = args.imgtype
	IMGFOLDER = args.imgfolder
	OUTPATH = os.path.join(args.data_root, args.out) 

	out = dict()

	with open(XML_LOCATION, "r") as f:
		root = ET.parse(f).getroot()
		# root = [root]
		w = float(root[0][0][0][0].get("width"))
		h = float(root[0][0][0][0].get("height"))
		calibration = get_calibration(root)
		fl_x = float(calibration[1].text)
		fl_y = fl_x
		k1 = float(calibration[4].text)
		k2 = float(calibration[5].text)
		p1 = float(calibration[7].text)
		p2 = float(calibration[8].text)
		cx = float(calibration[2].text) + w/2
		cy = float(calibration[3].text) + h/2
		camera_angle_x = math.atan(float(w) / (float(fl_x) * 2)) * 2
		camera_angle_y = math.atan(float(h) / (float(fl_y) * 2)) * 2
		aabb_scale = 16

		out.update({"camera_angle_x": camera_angle_x})
		out.update({"camera_angle_y": camera_angle_y})
		out.update({"fl_x": fl_x})
		out.update({"fl_y": fl_y})
		out.update({"k1": k1})
		out.update({"k2": k2})
		out.update({"p1": p1})
		out.update({"p2": p2})
		out.update({"cx": cx})
		out.update({"cy": cy})
		out.update({"w": w})
		out.update({"h": h})
		out.update({"aabb_scale": aabb_scale})
		frames = list()
		for frame in root[0][2]:
			current_frame = dict()
			if not len(frame):
				continue
			if(frame[0].tag != "transform"):
				continue
			
			imagePath = os.path.join(IMGFOLDER, "%s.%s" % (frame.get("label"), IMGTYPE))
			current_frame.update({"file_path": imagePath})
			current_frame.update({"sharpness":sharpness(imagePath)})
			matrix_elements = [float(i) for i in frame[0].text.split()]
			transform_matrix = np.array([[matrix_elements[0], matrix_elements[1], matrix_elements[2], matrix_elements[3]], [matrix_elements[4], matrix_elements[5], matrix_elements[6], matrix_elements[7]], [matrix_elements[8], matrix_elements[9], matrix_elements[10], matrix_elements[11]], [matrix_elements[12], matrix_elements[13], matrix_elements[14], matrix_elements[15]]])
			
			#swap axes
			transform_matrix = transform_matrix[[2,0,1,3],:]
			#reflect z and Y axes
			current_frame.update({"transform_matrix":matrixMultiply(matrixMultiply(transform_matrix, reflectZ()), reflectY())} )
			
			frames.append(current_frame)
		out.update({"frames": frames})

	out = central_point(out, args)


	with open(OUTPATH, "w") as f:
		json.dump(out, f, indent=4)
		