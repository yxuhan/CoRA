import os
import argparse
import torch


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str)
opt, _ = parser.parse_known_args()


interval = 10

raw_frame_root = os.path.join(opt.data_root, "raw_frames")
img_root = os.path.join(opt.data_root, "image")
os.makedirs(img_root, exist_ok=True)
sharp_info = torch.load(os.path.join(opt.data_root, "sharpness.pkl"))


img_pth_list = sorted(os.listdir(raw_frame_root))
num_img = len(img_pth_list)
left = 0
while left + 1 < num_img:
    right = min(left + interval, num_img)
    max_sharp = 0
    for i in range(left, right):
        cur_pth = img_pth_list[i]
        cur_sharp = sharp_info[cur_pth] 
        if cur_sharp > max_sharp:
            max_sharp = cur_sharp
            max_path = cur_pth
    os.system("cp %s %s" % (os.path.join(raw_frame_root, max_path), os.path.join(img_root, max_path)))

    left = right
