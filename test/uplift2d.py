#import source3d.src
#from source3d.src
from __future__ import print_function, absolute_import
import os
import sys
import time
import copy

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

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

index_alphapose={
    # Use 17 skeleton point
    "Nose": 0,
    "RShoulder": 6,
    "RElbow": 8,
    "RWrist": 10,
    "LShoulder": 5,
    "LElbow": 7,
    "LWrist": 9,
    "RHip": 12,
    "RKnee": 14,
    "RAnkle": 16,
    "LHip": 11,
    "LKnee": 13,
    "LAnkle": 15,
    "REye": 2,
    "LEye": 1,
    "REar": 4,
    "LEar": 3
}

index_mapping={
# Alpha Pose to Human 3.6M
"Hip": [20, 0],
"RHip": [12,1],
"RKnee": [14,2],
"RFoot": [16,3],
"LHip": [11,6],
"LKnee": [13,7],
"LFoot": [15,8],
"Spine": [19,12],
"Thorax": [18,13],
"Head": [17,15],
"LShoulder": [5,17],
"LElbow": [7,18],
"LWrist": [9,19],
"RShoulder": [6,25],
"RElbow": [8,26],
"RWrist": [10,27]
}

def data_converter(data):
    data=data['keypoints']
    keypoints=[]
    kp_score=[]
    for a in range (0,len(data)):
        score=[]
        if ((a+3)%3==0):
            keypoints.append(data[a])
            keypoints.append(data[a+1])
        elif((a+1)%3==0):
            score=data[a]
            kp_score.append(score)

    return keypoints



def count_head(alpha_pose):
    x = (alpha_pose[index_alphapose['LEar']*2]+alpha_pose[index_alphapose['REar']*2])/2
    y = (alpha_pose[index_alphapose['LEar']*2+1]+alpha_pose[index_alphapose['REar']*2+1])/2
    return x,y


def count_thorax(alpha_pose):
    x = (alpha_pose[index_alphapose['LShoulder']*2]+alpha_pose[index_alphapose['RShoulder']*2])/2
    y = (alpha_pose[index_alphapose['LShoulder']*2+1]+alpha_pose[index_alphapose['RShoulder']*2+1])/2
    return x,y


def count_spine(alpha_pose):
    hip_x,hip_y=count_hip(alpha_pose)
    thorax_x,thorax_y=count_thorax(alpha_pose)
    x = (hip_x+thorax_x)/2
    y = (hip_y+thorax_y)/2
    return x,y

def count_hip(alpha_pose):
    x = (alpha_pose[index_alphapose['LHip']*2]+alpha_pose[index_alphapose['RHip']*2])/2
    y = (alpha_pose[index_alphapose['LHip']*2+1]+alpha_pose[index_alphapose['RHip']*2+1])/2
    return x,y

def add_features(alpha_pose):
    #Count Head
    head_x,head_y=count_head(alpha_pose)
    alpha_pose=np.append(alpha_pose,(head_x,head_y))
    
    #Count Thorax
    thorax_x,thorax_y=count_thorax(alpha_pose)
    alpha_pose=np.append(alpha_pose,(thorax_x,thorax_y))
 
    
    #Count Spine
    spine_x,spine_y=count_spine(alpha_pose)
    alpha_pose=np.append(alpha_pose,(spine_x,spine_y))
    
    #Count Hip
    hip_x,hip_y=count_hip(alpha_pose)
    alpha_pose=np.append(alpha_pose,(hip_x,hip_y))
    
    return alpha_pose

def map_alpha_to_human(alpha_pose):
    alpha_pose=add_features(alpha_pose)
    temp_list = [None] * 64
    for a,b in index_mapping.items():
        temp_list[b[1]*2]  =  alpha_pose[b[0]*2]
        temp_list[b[1]*2+1]=alpha_pose[b[0]*2+1]
    human36m=np.asarray(temp_list)
    return human36m

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
print(">>> creating model")
model = LinearModel()
model = model.cuda()
model.apply(weight_init)
print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
criterion = nn.MSELoss(size_average=True).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

print(">>> loading ckpt from '{}'".format('source3d/checkpoint/test/ckpt_best.pth.tar'))
ckpt = torch.load(model_path)
start_epoch = ckpt['epoch']
err_best = ckpt['err']
glob_step = ckpt['step']
lr_now = ckpt['lr']
model.load_state_dict(ckpt['state_dict'])
optimizer.load_state_dict(ckpt['optimizer'])
print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))

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
        print(outputs.shape)
        outputs_unnorm = data_process.unNormalizeData(outputs.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])

        # remove dim ignored
        dim_use = np.hstack((np.arange(3), stat_3d['dim_use']))

        outputs_use = outputs_unnorm[:, dim_use]
        pred_result.append(outputs_unnorm)
        
        # update summary
        if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

        bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.6f}' \
            .format(batch=i + 1,
                    size=len(test_loader),
                    batchtime=batch_time * 10.0,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg)
        bar.next()
    bar.finish()
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

def inferencealphaposeto3D(path,fixing=False,save_npy=False):
    # Opening JSON file 
    f = open(path) 

    # returns JSON object as  
    # a dictionary 
    data = json.load(f) 


    # Closing file 
    f.close() 
    
    #Create Datatest for 2D to 3D Inference
    all_test_data=create_datatest(data)
    
    #Doing Inference
    pred_result_all=test(all_test_data, model, criterion, new_stat_3d) #All
    
    #Combine Prediction Result
    prediction_list=combine_prediction(pred_result_all)
    
    if fixing:
        #Fixing for unity
        test_2d_normalized = np.asarray(all_test_data.dataset.test_inp) 
        fixed=correct_3D(prediction_list,test_2d_normalized)
    else:
        fixed=prediction_list
    
    if save_npy:
        base=os.path.basename(path)
        base=os.path.splitext(base)[0]

        with open('../inference_result_npy/'+base+'.npy', 'wb') as f:
            np.save(f, fixed)
    
    return fixed

def save_to_json(result_3D, input_path, output_path):
    f = open(input_path) 
    dim_use = np.hstack((np.arange(3), dim_to_use_3d))

    # returns JSON object as  
    # a dictionary 
    data = json.load(f) 
    
    for a,b in zip(data,result_3D):
        a['keypoints']=b[dim_use].tolist()
        a['visualize']=b.tolist()
        
    with open(output_path, 'w') as fp:
        fp.write(json.dumps(data))
    
def save_to_json_2D(result_2D, input_path, output_path):
    f = open(input_path) 

    # returns JSON object as  
    # a dictionary 
    data = json.load(f) 
    
    for a,b in zip(data,result_2D):
        a['keypoints']=b[dim_to_use_2d].tolist()
        
    with open(output_path, 'w') as fp:
        fp.write(json.dumps(data))

## 3D After doing proscrutes to another angle


def find_transformation_3D(skeleton_a,skeleton_b,fullskeleton):
    
    # remove dim ignored
    dim_use = np.hstack((np.arange(3), new_stat_3d['dim_use']))

    gt = skeleton_a[dim_use]
    out = skeleton_b[dim_use]
    gt = gt.reshape(-1, 3)
    out = out.reshape(-1, 3)
    _, Z, T, b, c = get_transformation(gt, out, True)
    
    skeleton3D=[]
    for i,skeleton in enumerate(fullskeleton):
        skeleton = skeleton.reshape(-1, 3)
        skeleton = (b * skeleton.dot(T)) + c
        skeleton3D.append(skeleton)
    skeleton3D=np.asarray(skeleton3D)
    skeleton3D=skeleton3D.reshape(skeleton3D.shape[0],96)

    return  skeleton3D
    
def find_pair_transformation(path_1,pose_class_1,image_id_1,path_2,pose_class_2,image_id_2):
    with open(path_1, 'rb') as f:
        angle1 = json.load(f)
    with open(path_2, 'rb') as f:
        angle2 = json.load(f)
    
    angle1_temp=[]
    angle1_all=[]

    for a in angle1:
        if (a['pose_class']==pose_class_1) and (a['image_id']==image_id_1):
            angle1_temp.append(a)
    

    angle2_before=[]
    angle2_all=[]
    for a in angle2:
        if (a['pose_class']==pose_class_2) and (a['image_id']==image_id_2):
            angle2_before.append(a)
        angle2_all.append(a['visualize'])
    
    angle2_transformed_3D=find_transformation_3D(np.asarray(angle1_temp[0]['visualize']),
                                             np.asarray(angle2_before[0]['visualize']),
                                             np.asarray(angle2_all))
    

    return angle2_transformed_3D

def get_transformation(X, Y, compute_optimal_scale=True):
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA ** 2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    c = muX - b * np.dot(muY, T)

    return d, Z, T, b, c

## 2D Mapping after doing proscrutes to another angle

import h5py

def load_cameras(bpath='cameras.h5', subjects=None):
    """
    :param bpath: *.h5
    :param subjects:
    :return: (dict)
    """

    if subjects is None:
        subjects = [1, 5, 6, 7, 8, 9, 11]
    rcams = {}

    with h5py.File(bpath, 'r') as hf:
        for s in subjects:
            for c in range(4):  # There are 4 cameras in human3.6m
                a = load_camera_params(hf, 'subject%d/camera%d/{0}' % (s, c + 1))
                rcams[(s, c + 1)] = a

    return rcams


def map3dto2dcamera( poses_set, cams, ncams=4 ):
    """
    Project 3d poses using camera parameters

    cams: dictionary with camera parameters
    ncams: number of cameras per subject

    """

    for cam in range( ncams ):
        R, T, f, c, k, p, name = cams[ (11, cam+1) ]
        pts2d, _, _, _, _ = project_point_radial( np.reshape(poses_set, [-1, 3]), R, T, f, c, k, p )

        pts2d = np.reshape( pts2d, [-1, len(H36M_NAMES)*2] )

    return pts2d

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


def mapto2D(skeletons,rcams):
    """

    """
    skeleton2D=[]
    for i,skeleton in enumerate(skeletons):
        mapped=map3dto2dcamera(skeleton,rcams,4)
        skeleton2D.append(mapped)
    skeleton2D=np.asarray(skeleton2D)
    skeleton2D=skeleton2D.reshape(skeleton2D.shape[0],64)
    return skeleton2D

SUBJECT_IDS = [1,5,6,7,8,9,11]
this_file = os.path.dirname(os.path.realpath('__file__'))
rcams = cameras.load_cameras(os.path.join(this_file, "../data/h36m/metadata.xml"), SUBJECT_IDS)


