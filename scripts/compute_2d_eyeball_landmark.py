import os
import cv2
from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor
from ibug.face_alignment.utils import plot_landmarks
from tqdm import tqdm
import torch
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str, default="data/old_neutral")
opt, _ = parser.parse_known_args()


data_root = opt.data_root
img_root = os.path.join(data_root, "image")
save_path = os.path.join(data_root, "2d_eyeball_landmarks.pkl")
vis_root = os.path.join(data_root, "vis_landmarks")
os.makedirs(vis_root, exist_ok=True)


img_pth_list = [
    "00280.png", "00291.png", "00312.png", "00335.png", "00349.png",
]  # manually select some nearly-fronal view

face_detector = RetinaFacePredictor(
    threshold=0.8, device='cuda:0',
    model=RetinaFacePredictor.get_model('resnet50')
)

landmark_detector = FANPredictor(
    device='cuda:0', model=FANPredictor.get_model('2dfan2_alt')
)

info = {}

for pth in tqdm(img_pth_list):
    image = cv2.imread(os.path.join(img_root, pth))
    detected_faces = face_detector(image, rgb=False)
    landmarks, scores = landmark_detector(image, detected_faces, rgb=False)

    info[pth] = {
        "landmarks": landmarks[0],
        "scores": scores[0],
    }

    # Draw the landmarks onto the image
    for lmks, scs in zip(landmarks, scores):
        plot_landmarks(image, lmks, scs, threshold=0.2)

    cv2.imwrite(os.path.join(vis_root, pth), image)

torch.save(info, save_path)
