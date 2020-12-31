import torch.multiprocessing as mp
import os
import cv2
import numpy as np
import torch
import time
import sys
from loguru import logger
from easydict import EasyDict as edict

from actRec.F import single_normalize_min_
from actRec import models
from actRec.models import get_model
from demo.classifier_config.apis import get_classifier_cfg

class classifier():
    def __init__(self, opt):

        self.opt   = opt
        self.cfg   = get_classifier_cfg(self.opt)
        self.model = None
        self.holder = edict()

    def load_model(self):
        self.model = get_model(self.cfg.MODEL,self.cfg.tagI2W)
        ckpt = torch.load(self.cfg.CHEKPT, map_location=self.opt.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if len(self.opt.gpus) > 1:
            self.model = torch.nn.DataParallel(self.model
                , device_ids=self.opt.gpus).to(self.opt.device)
        else:
            self.model.to(self.opt.device)
        self.model.eval()
    
    def predict_action(self, keypoints):
        points = keypoints.numpy()
        points = single_normalize_min_(points)
        points = points.reshape(1,34)
        actres = self.model.exe(points,self.opt.device,self.holder)
        return actres

    def drawTagToImg(self,img, prediction):
        tag = self.cfg.tagI2W[prediction]
        img = cv2.putText(img, tag, (800,40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        return img

   