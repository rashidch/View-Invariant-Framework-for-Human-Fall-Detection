'''
Get 3D before rotation and after rotation
'''
from __future__ import print_function, absolute_import
import os
import sys
import numpy as np

#import modules for inference from uplift2d package
from test.uplift2dto3d.alpha_h36m import map_alpha_to_human_classification, data_converter,map_alpha_to_human
from test.uplift2dto3d.uplift2d import *

def inferencealphaposeto3d_one(alpha2d,input_type="json"):
    converted=[]
    # human2d=alpha2d.copy()
    # human3d=alpha2d.copy()
    
    if (input_type=="json"):
        convert=np.asarray(data_converter(alpha2d))
    else:
        convert=alpha2d
        
    a=dim_to_use_2d.tolist()
    a.insert(18,28)
    a.insert(19,29)
    dim_to_use_2d_nose=np.asarray(a)

    human36m_output=map_alpha_to_human_classification(convert)
    human36m_output=human36m_output.astype('float')
    human2d=human36m_output[dim_to_use_2d_nose]

    human36m_alpha_example=map_alpha_to_human(convert)
    normalized=normalize_single_data(human36m_alpha_example,data_mean_2d,data_std_2d,dim_to_use_2d)
    normalized=normalized.astype('float')
    converted.append(normalized)
    converted=np.asarray(converted) 
    
    test_loader = DataLoader(
        dataset=Human36M_testing(converted,True),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    #Doing Inference
    pred_result_all=test(test_loader, model, criterion, new_stat_3d) #All
    dim_use = np.hstack((np.arange(3), dim_to_use_3d))
    prediction=pred_result_all[0][0][dim_use]
    human3d=prediction
        
    return human3d,human2d