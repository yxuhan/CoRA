import cv2
import os
import argparse
from tqdm import tqdm
import torch


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str)
opt, _ = parser.parse_known_args()


def compute_sharpness(pth):
    '''
    estimate image sharpness
    bigger is better
    '''
    img = cv2.imread(pth, 0)
    lap = cv2.Laplacian(img, cv2.CV_8UC1, ksize=3)
    return cv2.mean(lap)[0]


data_root = opt.data_root
img_root = os.path.join(data_root, "raw_frames")
shap_save_path = os.path.join(data_root, "sharpness.pkl")
sharpness = {}
for pth in tqdm(sorted(os.listdir(img_root))):
    sharpness[pth] = compute_sharpness(os.path.join(img_root, pth))
torch.save(sharpness, shap_save_path)
