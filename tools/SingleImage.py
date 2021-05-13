 

import argparse
import torch
import os
import platform
import sys
import math
import time

import cv2
import numpy as np
from prediction import classifier 
from source.alphapose.utils.config import update_config
from source.scripts.demo_api import SingleImageAlphaPose 

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Single-Image Demo')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")                  
parser.add_argument('--save_img', default=True, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=True, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=True, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=True, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)
"""-----------------------------Classifier Options----------------------------"""
parser.add_argument('--classifier', dest='classmodel', type=str, default='dnnsingle9',
                    help='choose classifer model, defualt dnn model')


args = parser.parse_args()
cfg = update_config(args.cfg)

args.gpus = [int(args.gpus[0])] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'


def predict_image():
    outputpath = "examples/res/"
    if not os.path.exists(outputpath + '/vis'):
        os.mkdir(outputpath + '/vis')
    
    demo = SingleImageAlphaPose(args, cfg)
    pose_predictor = classifier(args)
    im_name = args.inputimg    # the path to the target image
    image = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB)
    orig_image = image.copy()
    pose = demo.process(im_name, image)
    img = orig_image     # or you can just use: img = cv2.imread(image)
    img = demo.vis(img, pose)   # visulize the pose result
    
    
    with torch.no_grad():
        out=[]
        pose_predictor.load_model()
        for human in pose['result']:
            actres = pose_predictor.predict_action(human['keypoints'])
            actres = actres.cpu().numpy().reshape(-1)
            model_pred = np.argmax(actres)
            img = pose_predictor.drawTagToImg(img=img, prediction=model_pred)

            if img is None:
                logger.error('img is None,can cause by video stream interrupted')
            cv2.imshow("AlphaPose Demo",img)
            k = cv2.waitKey(60) 

            out.append(actres)
    
        print(out)
    
    # write the result to json:
    cv2.imwrite(os.path.join(outputpath, 'vis', os.path.basename(im_name)), img)
    #result = [pose]
    #pose_tuple = demo.writeJson(result, outputpath, form=args.format, for_eval=args.eval)

if __name__ == "__main__":
    predict_image()
