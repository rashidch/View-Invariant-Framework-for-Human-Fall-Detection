from test.uplift2dto3d.get3d import inferencealphaposeto3d_one
#from test.uplift2dto3d.alpha_h36m import map_alpha_to_human, data_converter
from test.uplift2dto3d.uplift2d import *
import os
import json
import numpy as np

if __name__== '__main__':
    #path='dataset/SkeletonData/2dh3.6m/taoyuan_angle1_2D_Original.json'
    path='source3d/json_test/taoyuan_angle2.json'
    f = open(path) 
    data = json.load(f)
    print('data_0',len(data[0]['keypoints']))
    print('data_0',data[0]['keypoints'])  
    # How to use function
    output_3d,output_2d= inferencealphaposeto3d_one(data[0], input_type="json")
    print('Skeleon shape:',len(output_2d['keypoints']))
    #print('Skeleon:',output_3d) 
