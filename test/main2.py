
import argparse
import torch
import os
import time
from test.pred_frame import predictSeq1, predict2dFrame, predict2d3dFrame, predictSeq2
from source.alphapose.utils.config import update_config


"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Single-Image Demo')
parser.add_argument('--cfg', type=str, required=False, 
                    default='source/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=False, 
                    default='source/pretrained_models/fast_res50_256x192.pth',
                    help='checkpoint file name')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")                
parser.add_argument('--vis', default=True, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
parser.add_argument('--save_out', type=str, default='outputs/dnn2d_multicam/AngleA.avi',
                        help='Save display to video file.')
parser.add_argument('--cam', dest='inputvideo', help='video-name', 
                    default='examples/demo/test/Angle_A.mp4') 
parser.add_argument('--transform', default=False, action='store_true',
                    help='Do you want to transform the angle?')
parser.add_argument('--transform_file', dest='transfile', help='transformation-camera-file',
                    default='examples/transformation_file/trans_Angle_B.pickle')
parser.add_argument('--classifier', dest='classmodel', type=str, default='dnntiny',
                    help='choose classifer model, defualt dnn model')
     
args = parser.parse_args()
cfg  = update_config(args.cfg)

args.gpus = [int(args.gpus[0])] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")


if __name__ == '__main__':
    
    #set prediction type
    predType = '2d3d'

    if predType == 'predictSeq1':
        predictSeq1(args,cfg)
    elif predType=='predictSeq2':
        predictSeq2(args,cfg)    
    elif predType == '2d':
        predict2dFrame(args, cfg)
    elif predType == '2d3d':
        predict2d3dFrame(args,cfg)
   