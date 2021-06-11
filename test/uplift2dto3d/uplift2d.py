#import source3d.src
#from source3d.src
from __future__ import print_function, absolute_import
import os
import sys
import time
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

from source3d.src import cameras
from source3d.src import data_utils
from source3d.src import linear_model
from source3d.src import procrustes
from source3d.src import viz
import glob
import cdflib
sys.argv = sys.argv[:1]

import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from source3d.src import data_process as data_process

import json 
from source3d.src.model import LinearModel, weight_init
import torch.nn as nn
from source3d.src import utils as utils

sys.path.append(os.path.join(os.path.dirname('__file__'), "progress"))

from progress.bar import Bar as Bar

# Load Human3.6M Skeleton
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

#import modules for inference from uplift2d package
from test.uplift2dto3d.alpha_h36m import map_alpha_to_human_classification,data_converter

import pickle

tf.app.flags.DEFINE_string("action","All", "The action to train on. 'All' means all the actions")

# Directories
tf.app.flags.DEFINE_string("cameras_path","source3d/data/h36m/metadata.xml", "File with h36m metadata, including cameras")

FLAGS = tf.app.flags.FLAGS

# Initiate Function
SUBJECT_IDS = [1,5,6,7,8,9,11]
this_file = os.path.dirname(os.path.realpath('__file__'))

#Load metadata.xml camera
rcams = cameras.load_cameras(os.path.join(this_file, FLAGS.cameras_path), SUBJECT_IDS)

# Do 3D Prediction from Custom Video
## Using Created Statistic Dictionary
normalize=True
actions = data_utils.define_actions( FLAGS.action )
# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = [1,5,6,7,8]
TEST_SUBJECTS  = [9,11]

stat_3D = torch.load('source3d/data/stat_3d.pth.tar')
stat_2D = torch.load('source3d/data/stat_2d.pth.tar')

data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = stat_2D['mean'],stat_2D['std'],stat_2D['dim_ignore'],stat_2D['dim_use']
data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d = stat_3D['mean'],stat_3D['std'],stat_3D['dim_ignore'],stat_3D['dim_use']

#All the json data
#load json data using json,load(f)
#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Human36M_testing(Dataset):
    def __init__(self, skeleton,many=False):
        """
        :param actions: list of actions to use
        :param data_path: path to dataset
        :param use_hg: use stacked hourglass detections
        :param is_train: load train/test dataset
        """

        self.test_inp, self.test_out = [], []

        # loading data
        # load test data
       
        if many:
            num_f= skeleton.shape
            for i in range(num_f[0]):
                self.test_inp.append(skeleton[i])
        else:
            self.test_inp.append(skeleton)


    def __getitem__(self, index):
        inputs = torch.from_numpy(self.test_inp[index]).float()

        return inputs

    def __len__(self):
        return len(self.test_inp)
    
def normalize_single_data(data, data_mean, data_std, dim_to_use ):
    """Normalizes a dictionary of poses

    Args
    data: dictionary where values are
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dim_to_use: list of dimensions to keep in the data
    Returns
    data_out: dictionary with same keys as data, but values have been normalized
    """

    data= data[dim_to_use]
    mu = data_mean[dim_to_use]
    stddev = data_std[dim_to_use]
    data_out= np.divide( (data - mu), stddev )

    return data_out
def create_datatest(data):
    converted=[]
    for dat in data:
        convert=np.asarray(data_converter(dat))
        human36m_alpha_example=map_alpha_to_human(convert)
        normalized=normalize_single_data(human36m_alpha_example,data_mean_2d,data_std_2d,dim_to_use_2d)
        normalized=normalized.astype('float')
        converted.append(normalized)

    converted=np.asarray(converted) 
    test_loader = DataLoader(
        dataset=Human36M_testing(converted,True),
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    return test_loader

# Load Model
model_path='source3d/checkpoint/test/ckpt_best.pth.tar'
# create model
#print(">>> creating 3D model")
model = LinearModel()
model = model.cuda()
model.apply(weight_init)
#print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
criterion = nn.MSELoss(size_average=True).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

#print(">>> loading ckpt from '{}'".format('source3d/checkpoint/test/ckpt_best.pth.tar'))
ckpt = torch.load(model_path)
start_epoch = ckpt['epoch']
err_best = ckpt['err']
glob_step = ckpt['step']
lr_now = ckpt['lr']
model.load_state_dict(ckpt['state_dict'])
optimizer.load_state_dict(ckpt['optimizer'])
#print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))

new_stat_3d={}
new_stat_3d['mean']=data_mean_3d
new_stat_3d['std']=data_std_3d
new_stat_3d['dim_use']=dim_to_use_3d
new_stat_3d['dim_ignore']=dim_to_ignore_3d
    
def test(test_loader, model, criterion, stat_3d, procrustes=False):
    losses = utils.AverageMeter()
    model.eval()

    all_dist = []
    pred_result=[]
    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(test_loader))

    for i, inps in enumerate(test_loader):
        inputs = Variable(inps.cuda())
        
        with torch.no_grad():
            outputs = model(inputs)

        # calculate erruracy
        #print(outputs.shape)
        outputs_unnorm = data_process.unNormalizeData(outputs.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])

        # remove dim ignored
        dim_use = np.hstack((np.arange(3), stat_3d['dim_use']))

        outputs_use = outputs_unnorm[:, dim_use]
        pred_result.append(outputs_unnorm)
        
        # update summary
        if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()
    '''
        bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.6f}' \
            .format(batch=i + 1,
                    size=len(test_loader),
                    batchtime=batch_time * 10.0,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg)
        bar.next()
    bar.finish()
    '''
    return pred_result

# Combine prediction from each batch into one prediction
def combine_prediction(pred_result_all):
    prediction_list = []
    for pred in pred_result_all:
        for pre in pred:
            prediction_list.append(pre)
    prediction_list=np.asarray(prediction_list)
    return prediction_list


def correct_3D(poses3d_input,poses2d_normalized):
    _max = 0
    _min = 10000
    poses3d=np.copy(poses3d_input)
    
    spine_x = poses2d_normalized[0][24]
    spine_y = poses2d_normalized[0][25]
            
    
    for i in range(poses3d.shape[0]):

        for j in range(32):

            tmp = poses3d[i][j * 3 + 2]
            poses3d[i][j * 3 + 2] = poses3d[i][j * 3 + 1]
            poses3d[i][j * 3 + 1] = tmp
            if poses3d[i][j * 3 + 2] > _max:
                _max = poses3d[i][j * 3 + 2]
                print("_max: ",_max)
            if poses3d[i][j * 3 + 2] < _min:
                _min = poses3d[i][j * 3 + 2]
                print("_min: ",_min)

    for i in range(poses3d.shape[0]):
        for j in range(32):
            poses3d[i][j * 3 + 2] = _max - poses3d[i][j * 3 + 2] + _min
            poses3d[i][j * 3] += (spine_x - 630)
            poses3d[i][j * 3 + 2] += (500 - spine_y)

    return poses3d

def project_point_radial(P, R, T, f, c, k, p):
    """
    Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: 2x1 (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    Returns
    Proj: Nx2 points in pixel space
    D: 1xN depth of each point in camera space
    radial: 1xN radial distortion per point
    tan: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
    """

    # P is a matrix of 3-dimensional points
    assert len(P.shape) == 2
    assert P.shape[1] == 3

    N = P.shape[0]
    X = R.dot(P.T - T)  # rotate and translate
    XX = X[:2, :] / X[2, :]  # 2x16
    r2 = XX[0, :] ** 2 + XX[1, :] ** 2  # 16,

    radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, N)), np.array([r2, r2 ** 2, r2 ** 3]))  # 16,
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]  # 16,

    tm = np.outer(np.array([p[1], p[0]]).reshape(-1), r2)  # 2x16

    XXX = XX * np.tile(radial + tan, (2, 1)) + tm  # 2x16

    Proj = (f * XXX) + c  # 2x16
    Proj = Proj.T

    D = X[2, ]

    return Proj, D, radial, tan, r2


def find_centroid_single(skeletons):
    skeleton_test=skeletons.reshape(int(51/3),3)
    centroid=np.mean(skeleton_test, axis=0)
    
    return centroid

def rotate_single(origin, skeletons, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    joints = []
    skeleton_test=skeletons.reshape(int(51/3),3)
    angle_xy,angle_xz,angle_yz=math.radians(angle[0]),math.radians(angle[1]),math.radians(angle[2])

    for j,join in enumerate(skeleton_test):
        ox, oy,oz = origin[0],origin[1],origin[2]
        px, py,pz = join[0],join[1],join[2]

        #Rotation XY
        qx = ox + math.cos(angle_xy) * (px - ox) - math.sin(angle_xy) * (py - oy)
        qy = oy + math.sin(angle_xy) * (px - ox) + math.cos(angle_xy) * (py - oy)
        px, py,pz=qx,qy,pz

        #Rotation XZ
        qx = ox + math.cos(angle_xz) * (px - ox) - math.sin(angle_xz) * (pz - oz)
        qz = oz + math.sin(angle_xz) * (px - ox) + math.cos(angle_xz) * (pz - oz)
        px, py,pz=qx,py,qz

        #Rotation YZ
        qy = oy + math.cos(angle_yz) * (py - oy) - math.sin(angle_yz) * (pz - oz)
        qz = oz + math.sin(angle_yz) * (py - oy) + math.cos(angle_yz) * (pz - oz)
        px, py,pz=px,qy,qz

        joints.append(px)
        joints.append(py)
        joints.append(pz)

    rotated=np.asarray(joints)
    return rotated

