import argparse
import cv2
import torch
import os
import time
import pickle
from test.utils import ResizePadding

from test.fallDetectionModule import detectFall
from source.alphapose.utils.config import update_config


"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(prog="Fall Detection App",description="This program reads video frames and predicts fall",
                                epilog="Enjoy the program")
parser.add_argument("--cfg",type=str,required=False,default="source/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml",
                    help="Alphapose configure file name",)
parser.add_argument("--checkpoint",type=str,required=False,default="source/pretrained_models/fast_res50_256x192.pth",
                    help="checkpoint file name",)
parser.add_argument("--detector", dest="detector", help="detector name", default="yolo")
parser.add_argument("--vis", default=True, action="store_true", help="visualize image")
parser.add_argument("--showbox", default=False, action="store_true", help="visualize human bbox")
parser.add_argument("--profile", default=False, action="store_true", help="add speed profiling at screen output")
parser.add_argument("--min_box_area", type=int, default=0, help="min box area to filter out")
parser.add_argument("--gpus",type=str,dest="gpus",default="0",
    help="choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)")
parser.add_argument("--flip", default=False, action="store_true", help="enable flip testing")
parser.add_argument("-vis","--vis_fast", dest="vis_fast", help="use fast rendering", action="store_true", default=False)
parser.add_argument("--save_out", type=str, default="outputs/alph_dnnnet2d_angleA/Angle_F.avi", help="Save display to video file.")
parser.add_argument("-c","--cam", dest="inputvideo", help="video-name", default="examples/demo/test/Angle_B.mp4")
parser.add_argument("--transform", default=True, action="store_true", help="Do you want to transform the angle?056")
parser.add_argument("-tfile","--transform_file",dest="transfile",help="transformation-file",
                    default="examples/transformation_file/trans_Angle_F.pickle",)
parser.add_argument("-cls","--classifier", dest="classmodel", type=str, default="", 
                    help="choose classifer model, defualt dnn model")

args = parser.parse_args()
cfg = update_config(args.cfg)
args.gpus = [int(args.gpus[0])] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")

def main(args,cfg):

    n_frames = 1
    pose2d_size = 34
    pose3d_size = None
    humanData = torch.zeros([n_frames, pose2d_size])
    
    fps_time = 0
    frameIdx = 0
    framenumber = -1
    tagI2W = ["Fall", "Stand"]
    groundtruth = []
    prediction_result = []

    cam_source = args.inputvideo
    im_name = cam_source.split("/")[-1]
    if type(cam_source) is str and os.path.isfile(cam_source):
        # capture video file usign VideoCapture handler.
        cap = cv2.VideoCapture(cam_source)
        # get width and height of frame
        #frame_width = int(cap.get(3))
        #frame_height = int(cap.get(4))
    
    fallModule = detectFall(args,cfg,n_frames,pose2d_size,pose3d_size)
    handle = open("examples/demo/test/labels/" + im_name.split(".")[0] + ".pickle", "rb")
    dictlabel = pickle.load(handle)

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # convert frame to RGB and resize with aspect ratio
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            framenumber = framenumber + 1
            framenumber_ = "{:05d}".format(framenumber)
            #frame = ResizePadding(frame, 320, 448)
            skeleton = fallModule.getPose(image,im_name)
            if skeleton is None:
                continue

            if frameIdx != n_frames:
                humanData[frameIdx, :] = skeleton
                frameIdx += 1
            if frameIdx == n_frames:
                #print(frameIdx, humanData)
                
                index, confidence = fallModule.predictFall(humanData)
                action_name = tagI2W[index]
                if index==0 and confidence<0.9:
                    index= 1
                    action_name = tagI2W[index]
                elif index==1 and confidence<0.9:
                    index = 0
                    action_name = tagI2W[index]
            
                #print("Conf:{:0.2f},Predicted Label:{}".format(confidence, action_name))
                #print("Pred:{},Predicted Label:{}".format(index, action_name))

                # render predicted class names on video frames
                if action_name == "Fall":
                    frame_new = cv2.putText(
                        frame,
                        text="Falling",
                        org=(340, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75,
                        color=(0, 0, 255),
                        thickness=2,
                    )

                elif action_name == "Stand":
                    frame_new = cv2.putText(
                        frame,
                        text="Standing",
                        org=(340, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75,
                        color=(0,255, 0),
                        thickness=2,
                    )
                
                frame_new = cv2.putText(
                    frame_new,
                    text="FPS: %f" % (1 / (time.time() - fps_time)),
                    org=(10, 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75,
                    color=(255, 255, 255),
                    thickness=2,
                )

                fps_time = time.time()
                humanData[: n_frames - 1] = humanData[1:n_frames]
                frameIdx = n_frames - 1

                # set opencv window attributes
                window_name = "FallDetection"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 720, 512)
                # Show Frame.
                cv2.imshow(window_name, frame_new)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                """Find recall precision and f1 score"""
                try:
                    label = dictlabel[framenumber_]
                    groundtruth.append(1 if label == "Stand" else 0)
                    prediction_result.append(1 if action_name == "Stand" else 0)
                    print("Predicted Label:",action_name)
                    print("Ground Label:",label)
                except:
                    pass
        else:
            print("Cap could not read the frame")
            break
    
    fallModule.getScores(groundtruth,prediction_result)
    # Clear resource.
    cap.release()
    #out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(args,cfg)

    '''
    # set prediction type
    predType = "alpha2d"
    if predType == "alpha2d":
        predict2d(args, cfg)
    elif predType == "hmseq":
        predictSeq(args, cfg)
    elif predType == "2d":
        predict2dFrame(args, cfg)
    elif predType == "2d3d":
        predict2d3dFrame(args, cfg)
    '''