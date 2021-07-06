from time import process_time
import numpy as np
import torch
from easydict import EasyDict as edict
from fallModels.normalize import normalize_min_, normalize3d_min_
from fallModels.models import get_model
from test.classifier_config.apis import get_classifier_cfg


class classifier():
    def __init__(self, opt, n_frames, pose2d_size, pose3d_size):

        self.opt   = opt
        self.cfg   = get_classifier_cfg(self.opt)
        self.model = None
        self.holder = edict()
        self.pose2d_size = pose2d_size
        self.pose3d_size = pose3d_size
        self.n_frames = n_frames


    def load_model(self):
        self.model = get_model(self.cfg.MODEL,self.cfg.tagI2W, n_frames=self.n_frames,pose2d_size =self.pose2d_size, pose3d=51)
        ckpt  = torch.load(self.cfg.CHEKPT, map_location=self.opt.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.opt.device)
        self.model.eval()
    
    def predict_action(self, keypoints):
        #predict using this function if 2d data in alphapose format i.e., if pose2d_size=24
        points = keypoints.numpy()
        points = normalize_min_(points)
        #if self.cfg.MODEL[:3]=='dnnnet':
        points = points.reshape(1,self.pose2d_size)
        #else:
        #points = points.reshape(1,self.n_frames,self.pose2d_size)
        #print(points.shape)
        actres = self.model.exe(points,self.opt.device,self.holder)
        return actres[1]
    
    def predict_2d(self, human2d):
        #predict using this function if 2d data in h3.6m format i.e., if pose2d_size=34
        human2d = human2d.numpy()
        points = normalize_min_(human2d)
        # single frame
        if self.cfg.MODEL[:3]=='dnn':
            points = points.reshape(1,self.pose2d_size)
        else:
            #if sequence of frames
            points = points.reshape(1,self.n_frames,self.pose2d_size)
        actres = self.model.exe(points,self.opt.device,self.holder)
        return actres[1]

    def predict_2d3d(self, human2d, human3d):
        #predict using this function if 2d and 3d data in h3.6m format i.e., if pose2d=34 and pose3d=51
       
        points2d = normalize_min_(human2d)
        points3d = normalize3d_min_(human3d)
        #print('Skeleton 2d shape:',human2d.shape)
        #print('Skeleton :',human2d)
        #print('Skeleton 3d shape:',human3d.shape)
        #print('Skeleton:',human3d)
        
        if self.cfg.MODEL[:3]=='net':
            points2d = points2d.reshape(1,self.pose2d_size)
            points3d = points3d.reshape(1,self.pose3d_size)
        actres = self.model.exe(points2d, points3d,self.opt.device,self.holder)
        return actres[3]

