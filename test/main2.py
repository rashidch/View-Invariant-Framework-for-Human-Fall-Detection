import argparse
import torch
import os
import time
from test.pred_frame import predictSequence, predict2dFrame, predict2d3dFrame, predH36mSeq
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
parser.add_argument('--save_out', type=str, default='outputs/lstm2d/7.avi',
                    help='Save display to video file.')
parser.add_argument('--cam', dest='inputvideo', help='video-name',
                    default='examples/demo/test/7.mp4')

parser.add_argument('--classifier', dest='classmodel', type=str, default='fallmodel',
                    help='choose classifer model, defualt dnn model')

args = parser.parse_args()
cfg = update_config(args.cfg)

args.gpus = [int(args.gpus[0])] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")

if __name__ == '__main__':

    # set prediction type
    predType = '2d'
    if predType == 'seq':
        predictSequence(args, cfg)
    elif predType == 'seqh36m':
        predH36mSeq(args, cfg)
    elif predType == '2d':
        predict2dFrame(args, cfg)
    elif predType == '2d3d':
        predict2d3dFrame(args, cfg)