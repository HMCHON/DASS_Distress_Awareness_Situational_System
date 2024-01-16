'''
import  librarys
'''
from __future__ import print_function, division

import os
import cv2
import argparse
import torch
import warnings
import sys
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import time
import yaml
import math
import csv
from torchvision import datasets, transforms
from PIL import ImageFont, ImageDraw, Image

'''
system path append
'''
sys.path.append(os.path.join(os.path.dirname(__file__), 'thirdparty/fast-reid'))
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from detector import build_detector
from launch import exe

'''
parse args setting
'''
def parse_args_deepsort():
    # Deep sort
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str, default="test10.mp4", help="set your video direct") # VIDEO PATH
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--config_mmdetection", type=str, default="configs/mmdet.yaml")
    parser.add_argument("--config_detection", type=str, default="configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="configs/deep_sort.yaml")
    parser.add_argument("--config_fastreid", type=str, default="configs/fastreid.yaml")
    parser.add_argument("--fastreid", action="store_true", default="--fastreid")
    parser.add_argument("--mmdet", action="store_true") # mmdet have problem
    parser.add_argument("--display", action="store_true", default = "--display")
    parser.add_argument("--frame_interval", type=int, default=1.5)
    parser.add_argument("--display_width", type=int, default=1080)
    parser.add_argument("--display_height", type=int, default=720)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument("--person", action = "store", type = int, default = "2")

    args = parser.parse_args()

    return args





'''
execution
'''
if __name__ == "__main__":
    args = parse_args_deepsort()
    cfg = get_config()

    cfg.merge_from_file(args.config_detection)
    cfg.USE_MMDET = False # MMDET have problem in my computer
    cfg.merge_from_file(args.config_deepsort)

    if args.fastreid:
        cfg.merge_from_file(args.config_fastreid)
        cfg.USE_FASTREID = True
    else:
        cfg.USE_FASTREID = False

    exe.run()


    #with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        #vdo_trk.run()
