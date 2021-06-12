import torch
import os
import time

import cv2
import numpy as np
from test.prediction import classifier
from test.prediction import PoseEstimation
from test.vis import vis_frame
from test.utils import ResizePadding

from test.uplift2dto3d.get3d import inferencealphaposeto3d_one
from test.uplift2dto3d.uplift2d import *


def predictSequence(args, cfg):
    '''
        this function takes skeleton in alphapose format
    '''

    tagI2W = ["Fall", "Stand"]
    cam_source = args.inputvideo

    if type(cam_source) is str and os.path.isfile(cam_source):
        # capture video file usign VideoCapture handler.
        cap = cv2.VideoCapture(cam_source)
        # get width and height of frame
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

    output = args.save_out
    if not os.path.exists(output):
        os.makedirs(output)
    # create video writer handler
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))
    # get the name of video file
    im_name = cam_source.split('/')[-1]

    n_frames = 1
    fps_time = 0
    pose2d_size = 34
    pose3d_size = None
    humanData = torch.zeros([n_frames, pose2d_size])

    # create objects of pose esimator and classifier class
    pose_estimator = PoseEstimation(args, cfg)
    pose_predictor = classifier(args, n_frames, pose2d_size, pose3d_size)
    frameIdx = 0
    while (cap.isOpened()):

        ret, frame = cap.read()
        if ret == True:

            # convert frame to RGB and resize with aspect ratio
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = ResizePadding(frame, 320, 448)
            image = frame.copy()
            # get the body keypoints for current frame
            pose = pose_estimator.process(im_name, image)

            if pose == None:
                # cv2.imshow(window_name, image)
                continue

            # concatenate (N=5) frames to create a sequence of body keypoints
            if frameIdx != n_frames:
                human = pose['result'][0]
                humanData[frameIdx] = human['keypoints'][5:].view(1, pose2d_size)
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
                    if action_name == 'Fall':
                        frame_new = cv2.putText(frame, text='Falling', org=(340, 20),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 0, 0),
                                                thickness=2)

                    elif action_name == 'Stand':
                        frame_new = cv2.putText(frame, text='Standing', org=(340, 20),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 230, 0),
                                                thickness=2)

                    elif action_name == 'Tie':
                        frame = cv2.putText(frame, text='Tying', org=(340, 20),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 230, 0),
                                            thickness=2)

                    frame_new = cv2.putText(frame, text='FPS: %f' % (1 / (time.time() - fps_time)),
                                            org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale=0.75, color=(255, 255, 255), thickness=2)

                    frame_new = frame_new[:, :, ::-1]
                    fps_time = time.time()
                    humanData[:n_frames - 1] = humanData[1:n_frames]
                    frameIdx = n_frames

                    # set opencv window attributes
                    window_name = "Fall Detection Window"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 720, 512)
                    # Show Frame.
                    cv2.imshow(window_name, frame_new)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    dim = (int(frame_width), int(frame_height))
                    frame_new = cv2.resize(frame_new, dim, interpolation=cv2.INTER_AREA)
                    out.write(frame_new.astype('uint8'))
        else:
            break
    # Clear resource.
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def predH36mSeq(args, cfg):
    '''
        this function takes skeleton in h36m format and stack the frames to make sequence
    '''

    tagI2W = ["Fall", "Stand"]
    cam_source = args.inputvideo

    if type(cam_source) is str and os.path.isfile(cam_source):
        # capture video file usign VideoCapture handler.
        cap = cv2.VideoCapture(cam_source)
        # get width and height of frame
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

    output = args.save_out
    outdir = os.path.dirname(args.save_out)
    print(outdir)
    # if not os.path.exists(outdir):
    #   os.makedirs(outdir)
    # create video writer handler
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))
    # get the name of video file
    im_name = cam_source.split('/')[-1]

    n_frames = 5
    fps_time = 0
    pose2d_size = 34
    pose3d_size = None
    humanData = np.zeros([n_frames, pose2d_size])

    # create objects of pose esimator and classifier class
    pose_estimator = PoseEstimation(args, cfg)
    pose_predictor = classifier(args, n_frames, pose2d_size, pose3d_size)
    frameIdx = 0
    while (cap.isOpened()):

        ret, frame = cap.read()
        if ret == True:

            # convert frame to RGB and resize with aspect ratio
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = ResizePadding(frame, 320, 448)
            image = frame.copy()
            # get the body keypoints for current frame
            pose = pose_estimator.process(im_name, image)
            if pose == None:
                # cv2.imshow(window_name, image)
                continue
            # print('original pose',pose)
            _pose = pose['result'][0]['keypoints'].cpu().numpy().reshape(34, )
            # print("pose",_pose)
            human3d, human2d = inferencealphaposeto3d_one(_pose, input_type="array")
            # print('2d Skeleton shape:',human2d.shape)
            # print('3d Skeleton shape:',human3d.shape)

            # concatenate (N=5) frames to create a sequence of body keypoints
            if frameIdx != n_frames:
                print('frameIdx in B1-->', frameIdx)
                print('HumanData shape', humanData.shape)
                humanData[frameIdx, :] = human2d.reshape(1, pose2d_size)
                print('Human Data', humanData[frameIdx, :])
                frameIdx += 1

            # only predict the fall down action if the seqeunce lenght is 5
            if frameIdx == n_frames:
                print('frameIdx in B2-->', frameIdx)
                with torch.no_grad():
                    # load classifier model
                    pose_predictor.load_model()
                    # predict fall down action
                    actres = pose_predictor.predict_2d(humanData)
                    actres = actres.cpu().numpy().reshape(-1)
                    # print(actres)
                    predictions = np.argmax(actres)
                    # print(predictions)
                    # get confidence and fall down class name
                    confidence = round(actres[predictions], 3)
                    action_name = tagI2W[predictions]
                    # print("Confidence--> {},Action_Name-->{}".format(confidence, action_name))
                    frame = vis_frame(frame, pose, args)  # visulize the pose result
                    # render predicted class names on video frames
                    if action_name == 'Fall':
                        frame_new = cv2.putText(frame, text='Falling', org=(340, 20),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 0, 0),
                                                thickness=2)

                    elif action_name == 'Stand':
                        frame_new = cv2.putText(frame, text='Standing', org=(340, 20),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 230, 0),
                                                thickness=2)

                    elif action_name == 'Tie':
                        frame = cv2.putText(frame, text='Tying', org=(340, 20),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 230, 0),
                                            thickness=2)

                    frame_new = cv2.putText(frame, text='FPS: %f' % (1 / (time.time() - fps_time)),
                                            org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale=0.75, color=(255, 255, 255), thickness=2)

                    frame_new = frame_new[:, :, ::-1]
                    fps_time = time.time()
                    humanData[:n_frames - 1, :] = humanData[1:n_frames, :]
                    frameIdx = n_frames - 1

                    # set opencv window attributes
                    window_name = "Fall Detection Window"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 720, 512)
                    # Show Frame.
                    cv2.imshow(window_name, frame_new)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    dim = (int(frame_width), int(frame_height))
                    frame_new = cv2.resize(frame_new, dim, interpolation=cv2.INTER_AREA)
                    out.write(frame_new.astype('uint8'))
        else:
            break
    # Clear resource.
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def predict2dFrame(args, cfg):
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
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))
    # get the name of video file
    im_name = cam_source.split('/')[-1]

    n_frames = 1
    fps_time = 0
    pose2d_size = 34
    humanData = torch.zeros([n_frames, pose2d_size])

    # create objects of pose esimator and classifier class
    pose_estimator = PoseEstimation(args, cfg)
    pose_predictor = classifier(args, n_frames, pose2d_size)
    frameIdx = 0
    while (cap.isOpened()):

        ret, frame = cap.read()
        if ret == True:

            # convert frame to RGB and resize with aspect ratio
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = ResizePadding(frame, 320, 448)
            image = frame.copy()
            # get the body keypoints for current frame
            pose = pose_estimator.process(im_name, image)
            print('original pose', pose)
            _pose = pose['result'][0]['keypoints'].cpu().numpy().reshape(34, )
            print("pose", _pose)
            human3d, human2d = inferencealphaposeto3d_one(_pose, input_type="array")
            print('2d Skeleton shape:', human2d.shape)
            print('3d Skeleton shape:', human3d.shape)

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
                    if action_name == 'Fall':
                        frame_new = cv2.putText(frame, text='Falling', org=(340, 20),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 0, 0),
                                                thickness=2)

                    elif action_name == 'Stand':
                        frame_new = cv2.putText(frame, text='Standing', org=(340, 20),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 230, 0),
                                                thickness=2)

                    elif action_name == 'Tie':
                        frame = cv2.putText(frame, text='Tying', org=(340, 20),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 230, 0),
                                            thickness=2)

                    frame_new = cv2.putText(frame, text='FPS: %f' % (1 / (time.time() - fps_time)),
                                            org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale=0.75, color=(255, 255, 255), thickness=2)

                    frame_new = frame_new[:, :, ::-1]
                    fps_time = time.time()
                    humanData[:n_frames - 1] = humanData[1:n_frames]
                    frameIdx = 0

                    # set opencv window attributes
                    window_name = "Fall Detection Window"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 720, 512)
                    # Show Frame.
                    cv2.imshow(window_name, frame_new)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    dim = (int(frame_width), int(frame_height))
                    frame_new = cv2.resize(frame_new, dim, interpolation=cv2.INTER_AREA)
                    out.write(frame_new.astype('uint8'))
        else:
            break
    # Clear resource.
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def predict2d3dFrame(args, cfg):
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
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))
    # get the name of video file
    im_name = cam_source.split('/')[-1]

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
    while (cap.isOpened()):

        ret, frame = cap.read()
        if ret == True:

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
            _pose = pose['result'][0]['keypoints'].cpu().numpy().reshape(pose2d_size, )
            human3d, human2d = inferencealphaposeto3d_one(_pose, input_type="array")

            # merge the body keypints of (N=5) frames to create a sequence of body keypoints
            if frameIdx != n_frames:
                human = pose['result'][0]
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
                    if action_name == 'Fall':
                        frame_new = cv2.putText(frame, text='Falling', org=(340, 20),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 0, 0),
                                                thickness=2)

                    elif action_name == 'Stand':
                        frame_new = cv2.putText(frame, text='Standing', org=(340, 20),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 230, 0),
                                                thickness=2)

                    elif action_name == 'Tie':
                        frame = cv2.putText(frame, text='Tying', org=(340, 20),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 230, 0),
                                            thickness=2)

                    frame_new = cv2.putText(frame, text='FPS: %f' % (1 / (time.time() - fps_time)),
                                            org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale=0.75, color=(255, 255, 255), thickness=2)

                    frame_new = frame_new[:, :, ::-1]
                    fps_time = time.time()
                    humanData[:n_frames - 1] = humanData[1:n_frames]
                    frameIdx = 0

                    # set opencv window attributes
                    window_name = "Fall Detection Window"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 720, 512)
                    # Show Frame.
                    cv2.imshow(window_name, frame_new)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    dim = (int(frame_width), int(frame_height))
                    frame_new = cv2.resize(frame_new, dim, interpolation=cv2.INTER_AREA)
                    out.write(frame_new.astype('uint8'))
        else:
            break
    # Clear resource.
    cap.release()
    out.release()
    cv2.destroyAllWindows()