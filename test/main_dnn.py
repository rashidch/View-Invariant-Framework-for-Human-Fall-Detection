
import argparse
import torch
import os
import time

import cv2
import numpy as np
from test.prediction import classifier
from test.prediction import PoseEstimation 
from source.alphapose.utils.config import update_config
from test.vis import vis_frame, vis_frame_fast


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
parser.add_argument('--save_out', type=str, default='outputs/1.avi',
                        help='Save display to video file.')
parser.add_argument('--cam', dest='inputvideo', help='video-name', 
                    default='examples/demo/test/1.mp4') 

"""-----------------------------Classifier Options----------------------------"""
parser.add_argument('--classifier', dest='classmodel', type=str, default='dnntiny',
                    help='choose classifer model, defualt dnn model')

args = parser.parse_args()
cfg  = update_config(args.cfg)

args.gpus = [int(args.gpus[0])] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")


def ResizePadding(image, height, width):
    desired_size = (height, width)
    old_size = image.shape[:2]
    max_size_idx = old_size.index(max(old_size))
    ratio = float(desired_size[max_size_idx]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    if new_size > desired_size:
        min_size_idx = old_size.index(min(old_size))
        ratio = float(desired_size[min_size_idx]) / min(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

    image = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = desired_size[1] - new_size[1]
    delta_h = desired_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return image
    

def predict_frame():

    tagI2W = ["Fall","Stand"]
    cam_source = args.inputvideo

    if type(cam_source) is str and os.path.isfile(cam_source):
        # capture video file usign VideoCapture handler.
        cap = cv2.VideoCapture(cam_source)
        # get width and height of frame
        frame_width =  int(cap.get(3))
        frame_height = int(cap.get(4))
    
    output = args.save_out
    #create video writer handler 
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc('M','J','P','G'),20, (frame_width, frame_height))
    # get the name of video file
    im_name = cam_source.split('/')[-1]
    
    n_frames = 1
    fps_time = 0
    POSE_JOINT_SIZE = 24    
    #humanData = torch.zeros([n_frames, POSE_JOINT_SIZE])
    
    # create objects of pose esimator and classifier class
    pose_estimator = PoseEstimation(args, cfg) 
    pose_predictor = classifier(args, n_frames, POSE_JOINT_SIZE)
    frameIdx=0
    while (cap.isOpened()):
       
        ret, frame = cap.read()
        if ret ==True:

            # convert frame to RGB and resize with aspect ratio 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = ResizePadding(frame,320,448)
            image = frame.copy()
            # get the body keypoints for current frame 
            pose = pose_estimator.process(im_name, image)
            
            if pose==None:
                #cv2.imshow(window_name, image)
                continue
            
            #get keypoints data  
            human = pose['result'][0]    
            humanData = human['keypoints'][5:].view(1,POSE_JOINT_SIZE)
            
            #predict the fall down action 
            with torch.no_grad():
                # load classifier model
                pose_predictor.load_model()
                # predict fall down action
                actres = pose_predictor.predict_action(humanData)
                actres = actres.cpu().numpy().reshape(-1)
                #print(actres)
                predictions = np.argmax(actres)
                #print(predictions)
                #get confidence and fall down class name
                confidence = round(actres[predictions],3)
                action_name = tagI2W[predictions]
                print("Confidence: {},Action_Name:{}".format(confidence, action_name))
                frame = vis_frame(frame, pose, args)   # visulize the pose result
                
                # render predicted class names on video frames 
                if action_name=='Fall':
                    frame_new =  cv2.putText(frame, text='Falling', org=(340, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 0, 0), thickness=2)
                                    
                elif action_name=='Stand':
                    frame_new =  cv2.putText(frame, text='Standing', org=(340, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 230, 0), thickness=2)
                
                elif action_name=='Tie':
                    frame =  cv2.putText(frame, text='Tying', org=(340, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 230, 0), thickness=2)

                frame_new = cv2.putText(frame, text='FPS: %f' % (1 / (time.time() - fps_time)),
                                    org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=0.75, color=(255,255,255), thickness=2)

                frame_new = frame_new[:, :, ::-1]
                fps_time = time.time()
                #humanData[:n_frames-1] = humanData[1:n_frames]
                #frameIdx=n_frames
                
                #set opencv window attributes
                window_name = "Fall Detection Window" 
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name,720,512)
                # Show Frame.
                cv2.imshow(window_name, frame_new)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                dim = (int(frame_width),int(frame_height))
                frame_new = cv2.resize(frame_new,dim , interpolation = cv2.INTER_AREA)
                out.write(frame_new.astype('uint8'))
        else:
            break
    # Clear resource.
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    predict_frame()
   