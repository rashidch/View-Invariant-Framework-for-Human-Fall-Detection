from time import process_time
import numpy as np



import torch
from easydict import EasyDict as edict

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import cv2
from fallModels.normalize import normalize_min_, normalize3d_min_
from fallModels.models import get_model
from test.classifier_config.apis import get_classifier_cfg
from test.detection_loader import DetectionLoader

"""----- Load modules from source for structring code ----"""

from source.alphapose.utils.transforms import get_func_heatmap_to_coord
from source.alphapose.utils.pPose_nms import pose_nms
from source.alphapose.utils.transforms import flip, flip_heatmap
from source.alphapose.models import builder
from source.detector.apis import get_detector


class classifier():
    def __init__(self, opt, n_frames, pose2d_size, pose3d_size):

        self.opt = opt
        self.cfg = get_classifier_cfg(self.opt)
        self.model = None
        self.holder = edict()
        self.pose2d_size = pose2d_size
        self.pose3d_size = pose3d_size
        self.n_frames = n_frames

    def load_model(self):
        self.model = get_model(self.cfg.MODEL, self.cfg.tagI2W, n_frames=self.n_frames, pose2d_size=self.pose2d_size,
                               pose3d=self.pose3d_size)
        ckpt = torch.load(self.cfg.CHEKPT, map_location=self.opt.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.opt.device)
        self.model.eval()

    def predict_action(self, keypoints):
        # predict using this function if 2d data in alphapose format i.e., if pose2d_size=24
        points = keypoints.numpy()
        points = normalize_min_(points)
        if self.cfg.MODEL[:3] == 'dnn':
            points = points.reshape(1, self.pose2d_size)
        else:
            points = points.reshape(1, self.n_frames, self.pose2d_size)
        actres = self.model.exe(points, self.opt.device, self.holder)
        return actres[1]

    def predict_2d(self, human2d):
        # predict using this function if 2d data in h3.6m format i.e., if pose2d_size=34

        points = normalize_min_(human2d)
        # single frame
        if self.cfg.MODEL[:3] == 'dnn':
            points = points.reshape(1, self.pose2d_size)
        else:
            # if sequence of frames
            points = points.reshape(1, self.n_frames, self.pose2d_size)
        actres = self.model.exe(points, self.opt.device, self.holder)
        return actres[1]

    def predict_2d3d(self, human2d, human3d):
        # predict using this function if 2d and 3d data in h3.6m format i.e., if pose2d=34 and pose3d=51

        points2d = normalize_min_(human2d)
        points3d = normalize3d_min_(human3d)
        # print('Skeleton 2d shape:',human2d.shape)
        # print('Skeleton :',human2d)
        # print('Skeleton 3d shape:',human3d.shape)
        # print('Skeleton:',human3d)

        if self.cfg.MODEL[:3] == 'net':
            points2d = points2d.reshape(1, self.pose2d_size)
            points3d = points3d.reshape(1, self.pose3d_size)
        else:
            points = points.reshape(1, self.n_frames, self.pose2d_size)
        actres = self.model.exe(points2d, points3d, self.opt.device, self.holder)
        return actres[3]

    def predict_3d(self, keypoints):
        #predict using this function if 2d data in h3.6m format i.e., if pose2d_size=34
        points = normalize3d_min_(keypoints)
        #if self.cfg.MODEL[:3]=='dnn':
        if self.cfg.MODEL[:3]=='dnn':
            points = points.reshape(1,self.pose3d_size)
        else:
            points = points.reshape(1,self.n_frames,self.pose3d_size)
        actres = self.model.exe(points,self.opt.device,self.holder)
        return actres[1]


    def drawTagToImg(self, img, prediction):
        tag = self.cfg.tagI2W[prediction]
        img = cv2.putText(img, tag, (800, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        return img


class PoseEstimation():

    def __init__(self, args, cfg):

        self.args = args
        self.cfg = cfg

        # attributes for conveting hm_data to coords
        self.eval_joints = list(range(cfg.DATA_PRESET.NUM_JOINTS))
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        self.norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        self.hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE

        # Load pose model
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        print(f'Loading pose model from {args.checkpoint}...')
        self.pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        self.pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
        self.pose_model.to(args.device)
        self.pose_model.eval()
        # Human detection model for cropping the person
        self.det_loader = DetectionLoader(get_detector(self.args), self.cfg, self.args)

    def process(self, im_name, image):

        result = None
        try:
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = self.det_loader.process(im_name,
                                                                                                       image).read()
                if orig_img is None:
                    raise Exception("no image is given")

                if boxes is None or boxes.nelement() == 0:
                    return None

                # pose Estimation
                inps = inps.to(self.args.device)
                if self.args.flip:
                    inps = torch.cat((inps, flip(inps)))
                hm = self.pose_model(inps)
                if self.args.flip:
                    hm_flip = flip_heatmap(hm[int(len(hm) / 2):], self.pose_dataset.joint_pairs, shift=True)
                    hm = (hm[0:int(len(hm) / 2)] + hm_flip) / 2
                hm_data = hm.cpu()

                # convert hm_data to body key points along with their scores
                # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                assert hm_data.dim() == 4
                pose_coords = []
                pose_scores = []

                for i in range(hm_data.shape[0]):
                    bbox = cropped_boxes[i].tolist()
                    pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox,
                                                                   hm_shape=self.hm_size, norm_type=self.norm_type)
                    pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                    pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
                preds_img = torch.cat(pose_coords)
                preds_scores = torch.cat(pose_scores)

                boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                    pose_nms(boxes, scores, ids, preds_img, preds_scores, self.args.min_box_area)

                # filter boxes on based on highest score and consider a box with highest score only
                max_score = 0
                idx_max = 0
                for k, score in enumerate(scores):
                    if scores[k] > max_score:
                        max_score = scores[k]
                        idx_max = k

                _result = []
                k = idx_max
                _result.append(
                    {
                        'keypoints': preds_img[k],
                        'kp_score': preds_scores[k],
                        'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                        'idx': ids[k],
                        'bbox': [boxes[k][0], boxes[k][1], boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]]
                    }
                )
                result = {
                    'imgname': im_name,
                    'result': _result,
                    'pose_class': None
                }

        except Exception as e:
            print(repr(e))
            print('An error as above occurs when processing the images, please check it')
            pass
        except KeyboardInterrupt:
            print('===========================> Finish Model Running.')

        return result
