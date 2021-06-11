#import source3d.src
#from source3d.src
from __future__ import print_function, absolute_import

import numpy as np

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

sys.argv = sys.argv[:1]

import json

sys.path.append(os.path.join(os.path.dirname('__file__'), "progress"))

# Load Human3.6M Skeleton
import pickle

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
# "Nose": [14,0],
"Head": [17,15],
"LShoulder": [5,17],
"LElbow": [7,18],
"LWrist": [9,19],
"RShoulder": [6,25],
"RElbow": [8,26],
"RWrist": [10,27]
}

index_mapping_nose={
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
"Nose": [0,14],
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

#This function is not includding nose for 2D to 3D
def map_alpha_to_human(alpha_pose):
    alpha_pose=add_features(alpha_pose)
    temp_list = [None] * 64
    for a,b in index_mapping.items():
        temp_list[b[1]*2]=alpha_pose[b[0]*2]
        temp_list[b[1]*2+1]=alpha_pose[b[0]*2+1]
    human36m=np.asarray(temp_list)
    return human36m

#This function is includding nose for classification
def map_alpha_to_human_classification(alpha_pose):
    alpha_pose=add_features(alpha_pose)
    temp_list = [None] * 64
    for a,b in index_mapping_nose.items():
        temp_list[b[1]*2]=alpha_pose[b[0]*2]
        temp_list[b[1]*2+1]=alpha_pose[b[0]*2+1]
    human36m=np.asarray(temp_list)
    return human36m


def map_alpha_to_human_classification_json(path):
    # Opening JSON file 
    f = open(path) 
    converted=[]
    # returns JSON object as  
    # a dictionary 
    data = json.load(f) 

    for dat in data:
        convert=np.asarray(data_converter(dat))
        human36m_alpha_example=map_alpha_to_human_classification(convert)
        human36m_alpha_example=human36m_alpha_example.astype('float')
        converted.append(human36m_alpha_example)

    converted=np.asarray(converted) 
        
    # Closing file 
    f.close() 
    
    return converted