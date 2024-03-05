import os
import argparse
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str)
opt, _ = parser.parse_known_args()

mask_root = os.path.join(opt.data_root, "mask")
img_root = os.path.join(opt.data_root, "image")

for pth in tqdm(os.listdir(mask_root)):
    img_path = os.path.join(img_root, pth)
    mask_path = os.path.join(mask_root, pth)
    img = Image.open(img_path)
    mask = Image.open(mask_path)
    img = transforms.ToTensor()(img)
    mask = transforms.ToTensor()(mask)
    img = img * mask
    save_image(img, img_path)
