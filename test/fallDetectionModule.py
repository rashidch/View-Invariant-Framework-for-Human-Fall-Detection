import torch
import os
import time
import pickle
import cv2
import numpy as np
from test.classificationModule import classifier
from test.poseEstimationModule import PoseEstimation


from test.uplift2dto3d.get3d import inferencealphaposeto3d_one, transform3d_one
#from test.uplift2dto3d.uplift2d import *

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report

class detectFall():

    def __init__(self, args, cfg, n_frames, pose2d_size, pose3d_size):

        self.args = args
        self.cfg = cfg
        self.n_frames =  n_frames
        self.pose2d_size = pose2d_size
        self.pose3d_size = pose3d_size
        self.pose_estimator = PoseEstimation(self.args, self.cfg)
        self.fallDetector = classifier(self.args, self.n_frames, self.pose2d_size, self.pose3d_size)
        # load classifier model
        self.fallDetector.load_model()

    def getPose(self, image, im_name, format='alphapose'):
               
        # get the body keypoints for current frame
        poseDict = self.pose_estimator.process(im_name, image)
        if poseDict == None:
            return '_','zero','_'
        elif poseDict!= None:
            if format=='h36m':
                _pose = (poseDict["result"][0]["keypoints"].cpu().numpy().reshape(34,))
                if (self.args.transform):
                    self.loadTransformation()
                    print('<...Angle Transformation...>')
                    human3d_ = inferencealphaposeto3d_one(_pose, input_type="array", need_2d=False)
                    human3d, human2d = transform3d_one(self.trans, human3d_)
                else:
                    human3d, human2d = inferencealphaposeto3d_one(_pose, input_type="array", need_2d=True)
                return human2d.reshape(1,self.pose2d_size),poseDict,human3d
            elif format=='alphapose':
                human = poseDict["result"][0] 
                return human["keypoints"].reshape(1,self.pose2d_size), poseDict, '_'
    
    def loadTransformation(self):
        if (self.args.transform):
            trans_path = self.args.transfile
            with open(trans_path, 'rb') as fp:
                self.trans = pickle.load(fp)
        
    def predictFall(self,humanData, format='alphapose'):

        with torch.no_grad():
            if format=='alphapose':
                #predict fall down action
                actres = self.fallDetector.predict_action(humanData)
            elif format=='h36m':
                actres = self.fallDetector.predict_2d(humanData)

            actres = actres.cpu().numpy().reshape(-1)
            prediction = np.argmax(actres)
            #get confidence and class name
            confidence = round(actres[prediction], 3)
            #frame = vis_frame(frame, pose, args)  # visulize the pose result
            return prediction, confidence

    def getScores(self,groundtruth,prediction_result):
        
        tagI2W = ["Fall", "Stand"]
        precision = precision_score(np.asarray(groundtruth), np.asarray(prediction_result),pos_label=0,average="binary")
        recall = recall_score(np.asarray(groundtruth), np.asarray(prediction_result), pos_label=0,average="binary")
        f_score = f1_score(np.asarray(groundtruth), np.asarray(prediction_result), pos_label=0,average="binary")
        
        #precision_ = precision_score(np.asarray(groundtruth), np.asarray(prediction_result), average=None)
        #recall_ = recall_score(np.asarray(groundtruth), np.asarray(prediction_result), average=None)
        #f_score_ = f1_score(np.asarray(groundtruth), np.asarray(prediction_result), average=None)

        print("Precision : {:0.4f}".format(np.round(precision, 2)))
        print("Recall : {:0.4f}".format(np.round(recall, 4)))
        print("F1_Score: {:0.4f}".format(np.round(f_score, 4)))
        print("\n")
    
        #print("Precision per class : {}".format(np.round(precision_, 4)))
        #print("Recall per class : {}".format(np.round(recall_, 4)))
        #print("F1_Score per class: {}".format(np.round(f_score_, 4)))
        print(classification_report(np.asarray(groundtruth), np.asarray(prediction_result), target_names=tagI2W))


'''
def predict2dFrame(args, cfg):

    """
    1. This function takes 2d skeleton in h36m format.
    2. create a sequence of frames and input to dnn model
    """
    tagI2W = ["Fall", "Stand"]
    cam_source = args.inputvideo

    if type(cam_source) is str and os.path.isfile(cam_source):
        # capture video file usign VideoCapture handler.
        cap = cv2.VideoCapture(cam_source)
        # get width and height of frame
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

    output = args.save_out
    # create video writer handler
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 20, (frame_width, frame_height))
    # get the name of video file
    im_name = cam_source.split("/")[-1]

    n_frames = 1
    fps_time = 0
    pose2d_size = 34
    pose3d_size = 51
    humanData = torch.zeros([n_frames, pose2d_size])

    # create objects of pose esimator and classifier class
    pose_estimator = PoseEstimation(args, cfg)
    pose_predictor = classifier(args, n_frames, pose2d_size,pose3d_size = None)
    frameIdx = 0
    framenumber = -1
    groundtruth = []
    prediction_result = []

    with open("examples/demo/test/labels/" + im_name.split(".")[0] + ".pickle", "rb") as handle:
        dictlabel = pickle.load(handle)

    while cap.isOpened():

        ret, frame = cap.read()
        if ret == True:
            framenumber = framenumber + 1
            framenumber_ = "{:05d}".format(framenumber)
            print("Frame number: ", framenumber_)

            # convert frame to RGB and resize with aspect ratio
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = ResizePadding(frame, 320, 448)
            image = frame.copy()
            # get the body keypoints for current frame
            pose = pose_estimator.process(im_name, image)
            print("original pose", pose)
            _pose = (
                pose["result"][0]["keypoints"]
                .cpu()
                .numpy()
                .reshape(
                    34,
                )
            )
            print("pose", _pose)
            human3d, human2d = inferencealphaposeto3d_one(_pose, input_type="array")
            print("2d Skeleton shape:", human2d.shape)
            print("3d Skeleton shape:", human3d.shape)

            if pose == None:
                # cv2.imshow(window_name, image)
                continue

            # merge the body keypints of (N=5) frames to create a sequence of body keypoints
            if frameIdx != n_frames:
                humanData[frameIdx] = human2d.view(1, pose2d_size)
                frameIdx += 1

            # only predict the fall down action if the seqeunce lenght is 5
            if frameIdx == n_frames:

                with torch.no_grad():
                    # load classifier model
                    pose_predictor.load_model()
                    # predict fall down action
                    actres = pose_predictor.predict_action(humanData)
                    actres = actres.cpu().numpy().reshape(-1)
                    # print(actres)
                    predictions = np.argmax(actres)
                    # print(predictions)
                    # get confidence and fall down class name
                    confidence = round(actres[predictions], 3)
                    action_name = tagI2W[predictions]
                    print("Confidence: {},Action_Name:{}".format(confidence, action_name))
                    frame = vis_frame(frame, pose, args)  # visulize the pose result
                    # render predicted class names on video frames
                    if action_name == "Fall":
                        frame_new = cv2.putText(
                            frame,
                            text="Falling",
                            org=(340, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.75,
                            color=(255, 0, 0),
                            thickness=2,
                        )

                    elif action_name == "Stand":
                        frame_new = cv2.putText(
                            frame,
                            text="Standing",
                            org=(340, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.75,
                            color=(255, 230, 0),
                            thickness=2,
                        )

                    elif action_name == "Tie":
                        frame = cv2.putText(
                            frame,
                            text="Tying",
                            org=(340, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.75,
                            color=(255, 230, 0),
                            thickness=2,
                        )

                    frame_new = cv2.putText(
                        frame,
                        text="FPS: %f" % (1 / (time.time() - fps_time)),
                        org=(10, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75,
                        color=(255, 255, 255),
                        thickness=2,
                    )

                    frame_new = frame_new[:, :, ::-1]
                    fps_time = time.time()
                    humanData[: n_frames - 1] = humanData[1:n_frames]
                    frameIdx = 0

                    # set opencv window attributes
                    window_name = "Fall Detection Window"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 720, 512)
                    # Show Frame.
                    cv2.imshow(window_name, frame_new)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                    dim = (int(frame_width), int(frame_height))
                    frame_new = cv2.resize(frame_new, dim, interpolation=cv2.INTER_AREA)
                    out.write(frame_new.astype("uint8"))

                    """Find recall precision and f1 score"""
                    try:
                        label = dictlabel[framenumber_]
                        groundtruth.append(1 if label == "Stand" else 0)
                        prediction_result.append(1 if action_name == "Stand" else 0)
                        print("Ground Label is:", label)
                        print("Predicted Label is:", action_name)
                    except:
                        pass
        else:
            break
    
    precision = precision_score(np.asarray(groundtruth), np.asarray(prediction_result),pos_label=0,average="binary")
    recall = recall_score(np.asarray(groundtruth), np.asarray(prediction_result),pos_label=0, average="binary")
    f_score = f1_score(np.asarray(groundtruth), np.asarray(prediction_result),pos_label=0, average="binary")

    #precision_ = precision_score(np.asarray(groundtruth), np.asarray(prediction_result), average=None)
    #recall_ = recall_score(np.asarray(groundtruth), np.asarray(prediction_result), average=None)
    #f_score_ = f1_score(np.asarray(groundtruth), np.asarray(prediction_result), average=None)

    
    print("Precision : {}".format(np.round(precision, 2)))
    print("Recall : {}".format(np.round(recall, 2)))
    print("F1_Score: {}".format(np.round(f_score, 2)))
    print("\n")
    
    #print("Precision per class : {}".format(np.round(precision_, 4)))
    #print("Recall per class : {}".format(np.round(recall_, 4)))
    #print("F1_Score per class: {}".format(np.round(f_score_, 4)))
    print(classification_report(np.asarray(groundtruth), np.asarray(prediction_result), target_names=tagI2W))

    # Clear resource.
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def predict2d3dFrame(args, cfg):

    """
    1. This function takes 2d skeleton in alphapose format.
    2. create a sequence of frames and input to 2d3ddnn model
    """

    tagI2W = ["Fall", "Stand"]
    cam_source = args.inputvideo

    if (args.transform):
        trans_path = args.transfile
        with open(trans_path, 'rb') as fp:
            trans = pickle.load(fp)

    # get the name of video file
    im_name = cam_source.split("/")[-1]

    if type(cam_source) is str and os.path.isfile(cam_source):
        # capture video file usign VideoCapture handler.
        cap = cv2.VideoCapture(cam_source)
        # get width and height of frame
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

    output = args.save_out
    # create video writer handler
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 20, (frame_width, frame_height))
    # get the name of video file
    im_name = cam_source.split("/")[-1]

    n_frames = 1
    fps_time = 0
    pose2d_size = 34
    pose3d_size = 51
    humanData = np.zeros([n_frames, pose2d_size])
    humanData3d = np.zeros([n_frames, pose3d_size])

    # create objects of pose esimator and classifier class
    pose_estimator = PoseEstimation(args, cfg)
    pose_predictor = classifier(args, n_frames, pose2d_size, pose3d_size)
    frameIdx = 0
    framenumber = -1
    groundtruth = []
    prediction_result = []

    with open("examples/demo/test/labels/" + im_name.split(".")[0] + ".pickle", "rb") as handle:
        dictlabel = pickle.load(handle)

    while cap.isOpened():

        ret, frame = cap.read()
        if ret == True:

            framenumber = framenumber + 1
            framenumber_ = "{:05d}".format(framenumber)
            print("Frame number: ", framenumber_)

            # convert frame to RGB and resize with aspect ratio
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = ResizePadding(frame, 320, 448)
            image = frame.copy()
            # get the body keypoints for current frame
            pose = pose_estimator.process(im_name, image)
            if pose == None:
                # cv2.imshow(window_name, image)
                continue
            # convert 2d pose to 3d
            _pose = (
                pose["result"][0]["keypoints"]
                .cpu()
                .numpy()
                .reshape(
                    pose2d_size,
                )
            )
            if (args.transform):
                print('<...Angle Transformation...>')
                human3d_ = inferencealphaposeto3d_one(_pose, input_type="array", need_2d=False)
                human3d, human2d = transform3d_one(trans, human3d_)
            else:
                human3d, human2d = inferencealphaposeto3d_one(_pose, input_type="array", need_2d=True)

            # merge the body keypints of (N=5) frames to create a sequence of body keypoints
            if frameIdx != n_frames:
                humanData[frameIdx] = human2d.reshape(1, pose2d_size)
                humanData3d[frameIdx] = human3d.reshape(1, pose3d_size)
                frameIdx += 1

            # only predict the fall down action if the seqeunce lenght is 5
            if frameIdx == n_frames:

                with torch.no_grad():
                    # load classifier model
                    pose_predictor.load_model()
                    # predict fall down action
                    actres = pose_predictor.predict_2d3d(humanData, humanData3d)
                    actres = actres.cpu().numpy().reshape(-1)
                    # print(actres)
                    predictions = np.argmax(actres)
                    # print(predictions)
                    # get confidence and fall down class name
                    confidence = round(actres[predictions], 3)
                    action_name = tagI2W[predictions]
                    print("Confidence: {:.2f},Action_Name:{}".format(confidence, action_name))
                    frame = vis_frame(frame, pose, args)  # visulize the pose result
                    # render predicted class names on video frames
                    if action_name == "Fall":
                        frame_new = cv2.putText(
                            frame,
                            text="Falling",
                            org=(340, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.75,
                            color=(255, 0, 0),
                            thickness=2,
                        )

                    elif action_name == "Stand":
                        frame_new = cv2.putText(
                            frame,
                            text="Standing",
                            org=(340, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.75,
                            color=(255, 230, 0),
                            thickness=2,
                        )

                    elif action_name == "Tie":
                        frame = cv2.putText(
                            frame,
                            text="Tying",
                            org=(340, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.75,
                            color=(255, 230, 0),
                            thickness=2,
                        )

                    frame_new = cv2.putText(
                        frame,
                        text="FPS: %f" % (1 / (time.time() - fps_time)),
                        org=(10, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75,
                        color=(255, 255, 255),
                        thickness=2,
                    )

                    frame_new = frame_new[:, :, ::-1]
                    fps_time = time.time()
                    humanData[: n_frames - 1] = humanData[1:n_frames]
                    frameIdx = 0

                    # set opencv window attributes
                    window_name = "Fall Detection Window"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 720, 512)
                    # Show Frame.
                    cv2.imshow(window_name, frame_new)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                    dim = (int(frame_width), int(frame_height))
                    frame_new = cv2.resize(frame_new, dim, interpolation=cv2.INTER_AREA)
                    out.write(frame_new.astype("uint8"))

                    """Find recall precision and f1 score"""
                    try:
                        label = dictlabel[framenumber_]
                        groundtruth.append(1 if label == "Stand" else 0)
                        prediction_result.append(1 if action_name == "Stand" else 0)
                        print("Label is: ", label)
                        print("action_name is: ", action_name)
                    except:
                        pass
        else:
            break

    precision = precision_score(np.asarray(groundtruth), np.asarray(prediction_result), average="binary")
    recall = recall_score(np.asarray(groundtruth), np.asarray(prediction_result), average="binary")
    f_score = f1_score(np.asarray(groundtruth), np.asarray(prediction_result), average="binary")

    #cm = confusion_matrix(np.asarray(groundtruth), np.asarray(prediction_result))
    #cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    #accuracy_ = cm.diagonal()
    precision_ = precision_score(np.asarray(groundtruth), np.asarray(prediction_result), average=None)
    recall_ = recall_score(np.asarray(groundtruth), np.asarray(prediction_result), average=None)
    f_score_ = f1_score(np.asarray(groundtruth), np.asarray(prediction_result), average=None)

    # print('{} : Confusion Matrix: {}'.format(phase, conf_mat))
    print("Precision : {}".format(np.round(precision, 4)))
    print("Recall : {}".format(np.round(recall, 4)))
    print("F1_Score: {}".format(np.round(f_score, 4)))
    print("\n")
    #print("Accuracy per class : {}".format(np.round(accuracy_, 4)))
    print("Precision per class : {}".format(np.round(precision_, 4)))
    print("Recall per class : {}".format(np.round(recall_, 4)))
    print("F1_Score per class: {}".format(np.round(f_score_, 4)))
    print(classification_report(np.asarray(groundtruth), np.asarray(prediction_result), target_names=tagI2W))

    # Clear resource.
    cap.release()
    out.release()
    cv2.destroyAllWindows()
'''