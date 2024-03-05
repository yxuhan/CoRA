import os
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str)
opt, _ = parser.parse_known_args()


selected = os.listdir(os.path.join(opt.data_root, "debug"))
image_root = os.path.join(opt.data_root, "image")
seg_root = os.path.join(opt.data_root, "mask")
mat_root = os.path.join(opt.data_root, "mats")
hair_root = os.path.join(opt.data_root, "hair_mask")

for pth in tqdm(os.listdir(image_root)):
    if pth in selected:
        continue
    os.system("rm %s" % os.path.join(image_root, pth))
    os.system("rm %s" % os.path.join(seg_root, pth))
    os.system("rm %s" % os.path.join(mat_root, pth))
    os.system("rm %s" % os.path.join(hair_root, pth))
