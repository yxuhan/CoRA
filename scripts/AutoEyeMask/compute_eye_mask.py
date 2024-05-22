import os
import argparse
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from model import BiSeNet
import cv2


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str, default="/root/autodl-tmp/CoRA/data/old_neutral")
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--weight_path', type=str, default="79999_iter.pth")
opt, _ = parser.parse_known_args()


# the meaning of parsing classes
# https://github.com/zllrunning/face-parsing.PyTorch/issues/12
# atts = [0 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
# 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
mask_net = BiSeNet(n_classes=19)
mask_net.to(opt.device)
mask_net.load_state_dict(torch.load(opt.weight_path))
mask_net.eval()
mask_net_transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


data_root = opt.data_root
image_root = os.path.join(data_root, "image")
leye_mask_root = os.path.join(data_root, "leye_mask")
reye_mask_root = os.path.join(data_root, "reye_mask")
os.makedirs(leye_mask_root, exist_ok=True)
os.makedirs(reye_mask_root, exist_ok=True)

for pth in tqdm(os.listdir(image_root)):
    img_pth = os.path.join(image_root, pth)
    frame_tensor = transforms.ToTensor()(Image.open(img_pth))  # [4,h,w]
    h, w = frame_tensor.shape[1:]
    
    pad_size = (h - w) // 2
    pad_left = pad_size
    pad_right = h - w - pad_left
    frame = torch.clone(frame_tensor)
    left = torch.zeros(3, frame.shape[1], pad_left)
    right = torch.zeros(3, frame.shape[1], pad_right)
    frame = torch.cat([left, frame, right], dim=-1)
    frame_size = frame.shape[-1]
    frame512 = transforms.Resize((512, 512))(frame)

    # face parsing
    frame_input = mask_net_transform(frame512).to(opt.device).unsqueeze(0)
    out = mask_net(frame_input)[0][0].cpu()  # [19,h,w]
    parsing = torch.argmax(out, dim=0)  # [h,w]
    leye_mask = (parsing == 5).float()
    reye_mask = (parsing == 4).float()
    
    # translate it back to the origin image size
    leye_mask = leye_mask[None, ...].repeat(3, 1, 1)  # [3,h,w]
    leye_mask = transforms.Resize((frame_size, frame_size))(leye_mask)
    reye_mask = reye_mask[None, ...].repeat(3, 1, 1)  # [3,h,w]
    reye_mask = transforms.Resize((frame_size, frame_size))(reye_mask)

    leye_mask = leye_mask[..., pad_left:-pad_right]
    reye_mask = reye_mask[..., pad_left:-pad_right]
    
    leye_save_path = os.path.join(leye_mask_root, pth)
    reye_save_path = os.path.join(reye_mask_root, pth)

    save_image(leye_mask, leye_save_path)
    save_image(reye_mask, reye_save_path)
