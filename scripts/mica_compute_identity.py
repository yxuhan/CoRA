import argparse
import cv2
import os


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img_path', type=str)
parser.add_argument('--data_root', type=str)
parser.add_argument('--mica_root', type=str)
opt, _ = parser.parse_known_args()


mica_root = opt.mica_root
cur_root = os.getcwd()


frame_save_root = os.path.join(cur_root, opt.data_root, "mica_frame")
recon_save_root = os.path.join(cur_root, opt.data_root, "mica_recon")
os.makedirs(frame_save_root)
os.system("cp %s %s" % (opt.img_path, os.path.join(frame_save_root, "frame.png")))

os.chdir(mica_root)
os.system("python demo.py -i %s -o %s" % (frame_save_root, recon_save_root))
os.system(
    "cp %s %s" % (
        os.path.join(recon_save_root, "frame", "identity.npy"), 
        os.path.join(cur_root, opt.data_root),
    )
)
os.system("rm -rf %s" % frame_save_root)
os.system("rm -rf %s" % recon_save_root)
