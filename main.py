
import argparse
import torch
import os
import sys
import time

import cv2
import numpy as np
from demo.prediction import classifier 
from source.alphapose.utils.config import update_config
from source.scripts.demo_api import SingleImageAlphaPose  
from demo.camera import CamLoader

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Single-Image Demo')
parser.add_argument('--cfg', type=str, required=True, 
default='source/configs/coco/resnet/256x192_res50_lr-3_1x.yaml',
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--image', dest='inputimg',
                    help='image-name')                  
parser.add_argument('--save_img', default=True, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=True, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
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
parser.add_argument('--save_out', type=str, default='./outputs/1.avi',
                        help='Save display to video file.')
parser.add_argument('--cam', dest='inputvideo', help='video-name', 
                    default='/examples/demo/1.mp4') 
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


def ResizePadding(image, height, width):
    desized_size = (height, width)
    old_size = image.shape[:2]
    max_size_idx = old_size.index(max(old_size))
    ratio = float(desized_size[max_size_idx]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    if new_size > desized_size:
        min_size_idx = old_size.index(min(old_size))
        ratio = float(desized_size[min_size_idx]) / min(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

    image = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = desized_size[1] - new_size[1]
    delta_h = desized_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return image
    

def predict_frame():

    tagI2W = ["Fall","Stand", "Tie"]
    
    cam_source = args.inputvideo
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader(cam_source, queue_size=1000, output=args.save_out).start()
    
    im_name = cam_source.split('/')[-1]
    demo = SingleImageAlphaPose(args, cfg)
    pose_predictor = classifier(args)
    
    fps_time = 0
    f = 0
    count_fall_frames = 0
    flag_show_frames = 0
    while cam.grabbed():
        f += 1
        frame = cam.getitem()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = ResizePadding(frame,576,720)  
        image = frame.copy()
       
        pose = demo.process(im_name, image)    
        if pose is not None:
            frame = demo.vis(frame, pose)   # visulize the pose result

        window_name = "Fall Detection Window" 
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        #rame2 = frame.copy()
        if pose==None:
            #cv2.imshow(window_name, image)
            continue
        
        with torch.no_grad():
            pose_predictor.load_model()
            human = pose['result'][0]
            #print(human)
            actres = pose_predictor.predict_action(human['keypoints'])
            actres = actres.cpu().numpy().reshape(-1)
            predictions = np.argmax(actres)
            confidence = round(actres[predictions],3)
            print(confidence)
            action_name = tagI2W[predictions]

            if action_name=='Fall' and confidence>0.9:
                frame_new =  cv2.putText(frame, text='Falling', org=(520, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
                
                frame_new =  cv2.putText(frame, text='Confidence: {}'.format(confidence), org=(520, 60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 255, 255), thickness=2)
                   
                                
            elif action_name=='Stand' and confidence>0.9:
                frame_new =  cv2.putText(frame, text='Standing', org=(520, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 230, 0), thickness=2)
                frame_new =  cv2.putText(frame, text='Confidence: {}'.format(confidence), org=(520, 60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 255, 255), thickness=2)
            
            elif action_name=='Tie' and confidence>0.9:
                frame =  cv2.putText(frame, text='Tying', org=(520, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 230, 0), thickness=2)

                frame_new =  cv2.putText(frame, text='Confidence: {}'.format(confidence), org=(520, 60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 255, 255), thickness=2)
            
            frame_new = cv2.putText(frame, text='Frame: %d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                                org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1, color=(255,255,255), thickness=2)
            
            frame_new = frame_new[:, :, ::-1]
            fps_time = time.time()

            # Show Frame.
            cv2.resizeWindow('image', 600,600)
            cv2.imshow(window_name, frame_new)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            dim = (int(cam.frame_width),int(cam.frame_height))
            frame_new = cv2.resize(frame_new,dim , interpolation = cv2.INTER_AREA)
            cam.save_video(frame_new)

    # Clear resource.
    cam.stop()
    cam.out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    predict_frame()