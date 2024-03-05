import os
import argparse
import cv2
import glob
from moviepy.editor import ImageSequenceClip
from PIL import Image
import torch
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import facer


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str)
parser.add_argument('--device', type=str, default="cuda:0")
opt, _ = parser.parse_known_args()


data_root = opt.data_root
device = opt.device

# obtain region parsing maps
mat_root = os.path.join(data_root, "mats")
face_mask_root = os.path.join(data_root, "mask")
hair_mask_root = os.path.join(data_root, "hair_mask")
debug_root = os.path.join(data_root, "debug")
img_root = os.path.join(opt.data_root, "image")

debug_root = os.path.join(data_root, "debug")
os.makedirs(face_mask_root, exist_ok=True)
os.makedirs(hair_mask_root, exist_ok=True)
os.makedirs(debug_root, exist_ok=True)

img_path_list = sorted(glob.glob(os.path.join(img_root, "*.png")))

# set face parser
face_detector = facer.face_detector('retinaface/mobilenet', device=device)
face_parser = facer.face_parser('farl/lapa/448', device=device) # optional "farl/celebm/448"

for img_path in tqdm(img_path_list):
    try:
        image = facer.hwc2bchw(facer.read_hwc(img_path)).to(device=device)  # image: 1 x 3 x h x w
        with torch.inference_mode():
            faces = face_detector(image)
            faces = face_parser(image, faces)
        
        seg_logits = faces['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)[0]  # nclasses x h x w
        vis_seg_probs = seg_probs.argmax(dim=0)

        hair_mask = (vis_seg_probs == 10).float()
        face_mask = (vis_seg_probs >= 1).float()

        img_name = os.path.basename(img_path)
        mat = transforms.ToTensor()(Image.open(os.path.join(mat_root, img_name)))[-1].to(hair_mask.device)

        hair_mask = hair_mask * mat
        face_mask = face_mask * mat

        save_image(face_mask, os.path.join(face_mask_root, img_name))
        save_image(hair_mask, os.path.join(hair_mask_root, img_name))
        
        img = transforms.ToTensor()(Image.open(img_path)).to(hair_mask.device)
        face_mask = face_mask.expand_as(img)
        hair_mask = hair_mask.expand_as(img)
        row1 = torch.cat([face_mask, face_mask * img, hair_mask, hair_mask * img, img], dim=-1)
        save_image(row1, os.path.join(debug_root, img_name))
    
    except:
        continue
