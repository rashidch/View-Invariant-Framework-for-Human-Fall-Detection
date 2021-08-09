from test.uplift2dto3d.get3d import inferencealphaposeto3d_one
from test.uplift2dto3d.alpha_h36m import map_alpha_to_human_classification_json
from test.uplift2dto3d.uplift2d import *
import os
import json
import numpy as np

def save_to_json_original2D(result_2D, input_path, output_path):
    f = open(input_path) 

    # returns JSON object as  
    # a dictionary 
    data = json.load(f) 
    
    a=dim_to_use_2d.tolist()
    a.insert(18,28)
    a.insert(19,29)
    dim_to_use_2d_nose=np.asarray(a)

    for a,b in zip(data,result_2D):
        a['keypoints']=b[dim_to_use_2d_nose].tolist()
        
    with open(output_path, 'w') as fp:
        fp.write(json.dumps(data))



if __name__== '__main__':
    #path='dataset/SkeletonData/2dh3.6m/taoyuan_angle1_2D_Original.json'
    #path='source3d/json_test/taoyuan_angle2.json'
    path = 'dataset/SkeletonData/multicamTest.json'

    f = open(path) 
    data = json.load(f)
    print('data_0',len(data[0]['keypoints']))
    print('data_0',data[0]['keypoints'])  
    # How to use function
    #output_3d,output_2d=inferencealphaposeto3d_one(data[0], input_type="json")
    converted = map_alpha_to_human_classification_json(path)
    directory = os.path.dirname(path)
    
    output_path = os.path.join(directory,'multicamTest_h36m.json')
    mapping_result=map_alpha_to_human_classification_json(path)
    save_to_json_original2D(mapping_result, path, output_path)
    
    
