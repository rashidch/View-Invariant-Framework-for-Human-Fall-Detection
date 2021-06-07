'''
Get 3D before rotation and after rotation
'''
from __future__ import print_function, absolute_import

#import modules for inference from uplift2d package
from test.uplift2dto3d.alpha_h36m import map_alpha_to_human
from test.uplift2dto3d.uplift2d import *

def inferencealphaposeto3d_one(alpha2d,input_type="json",need_2d=True):
    converted=[]

    if (input_type=="json"):
        convert=np.asarray(data_converter(alpha2d))
    else:
        convert=alpha2d
        
    a=dim_to_use_2d.tolist()
    a.insert(18,28)
    a.insert(19,29)
    dim_to_use_2d_nose=np.asarray(a)

    if need_2d:
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

    # Doing Inference
    pred_result_all = test(test_loader, model, criterion, new_stat_3d)  # All
    dim_use = np.hstack((np.arange(3), dim_to_use_3d))
    prediction = pred_result_all[0][0][dim_use]
    centroids = find_centroid_single(prediction)
    angles = [0, 0, 260]
    human3d = rotate_single(centroids, prediction, angles)
    human3d = human3d - np.tile(human3d[:3], [17])



    if need_2d:
        return human3d,human2d
    else:
        return human3d



def map3dto2dcamera_single(poses_set, cams, ncams=4):
    """
    Project 3d poses using camera parameters

    cams: dictionary with camera parameters
    ncams: number of cameras per subject

    """

    for cam in range(ncams):
        R, T, f, c, k, p, name = cams[(11, cam + 1)]
        pts2d, _, _, _, _ = project_point_radial(np.reshape(poses_set, [-1, 3]), R, T, f, c, k, p)

        pts2d = np.reshape(pts2d, [17 * 2])

    return pts2d


def transform3d_one(trans, skeleton3d):
    T = trans['T']
    b = trans['b']
    c = trans['c']

    skeleton3d = skeleton3d.reshape(-1, 3)
    skeleton3d = (b * skeleton3d.dot(T)) + c
    poses3d = skeleton3d.reshape(51, )

    # Recenter the skeleton and put hip to coordinate 0,0,0
    poses3d = poses3d - np.tile(poses3d[:3], [17])
    poses2d = map3dto2dcamera_single(poses3d, rcams, 4)

    return poses3d, poses2d