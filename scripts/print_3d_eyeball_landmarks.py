from PIL import Image
import argparse
import torch
from torchvision import transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ldm_img_path', type=str)
parser.add_argument('--coord_path', type=str)
opt, _ = parser.parse_known_args()


img = transforms.ToTensor()(Image.open(opt.ldm_img_path))[:3]

idx_list = ((img[1] == 1) * (img[0] == 0) * (img[2] == 0)).nonzero()

for idx in idx_list:
    y, x = idx[0].item(), idx[1].item()
    img[:, y, x] = torch.tensor([0., 1., 0.])

coord = torch.load(opt.coord_path)[0]
for idx in idx_list:
    y, x = idx[0].item(), idx[1].item()
    print(coord[:, y, x].tolist())
