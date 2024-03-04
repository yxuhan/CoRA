import cv2
import os
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--video_path', type=str)
parser.add_argument('--data_root', type=str)
parser.add_argument('--step_size', default=1, type=int)
opt, _ = parser.parse_known_args()

video_path = opt.video_path
save_root = os.path.join(opt.data_root, "raw_frames")
os.makedirs(save_root, exist_ok=True)


# dump frames
cap = cv2.VideoCapture(video_path)
fid = 0
while True:
    _, frame = cap.read()
    if frame is None:
        break
    fid += 1
    if fid % opt.step_size == 0:
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # for MOV video
        frame = cv2.resize(frame, (720, 960))
        cv2.imwrite(os.path.join(save_root, "%05d.png" % fid), frame)
