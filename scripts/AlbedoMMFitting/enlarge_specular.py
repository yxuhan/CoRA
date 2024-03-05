import cv2
import torch
import os
import numpy as np
import argparse
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import kornia


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str)
opt, _ = parser.parse_known_args()


def post_process(spec_path, mask_path, save_path):
    spec_img = cv2.imread(spec_path)
    mask_img = transforms.ToTensor()(Image.open(mask_path))[None, ...]
    mask_dilate = kornia.morphology.dilation(mask_img, torch.ones(7, 7))
    mask_np = (mask_dilate[0].permute(1, 2, 0).mean(dim=-1, keepdim=True) == 1.).float().numpy().astype(np.uint8) * 255
    spec_inpaint = cv2.inpaint(spec_img, mask_np, 3 ,cv2.INPAINT_TELEA)
    cv2.imwrite(save_path, spec_inpaint)


for pth in tqdm(os.listdir(os.path.join(opt.data_root, "specular_bfm"))):
    spec_path = os.path.join(opt.data_root, "specular_bfm", pth)
    inpaint_path = os.path.join(opt.data_root, "inpaint_mask", pth)
    post_process(spec_path, inpaint_path, spec_path)
