import pandas as pd
import os 
import numpy as np

# read dataset from csv file
def Read2dPoseData(n_frames):
    
    # get csv file path
    curr_dir = os.getcwd()
    csv_file_path = os.path.join(curr_dir, 'dataset/DataCSV/taoyuan.csv')
    
    # list for storing data and labels
    data  = []
    label = []
    
    # lenth of sequence
    #n_frames = SinglePoseDataset.n_frames
    
    # read csv file
    KP_df = pd.read_csv(csv_file_path)
    #print("DataFrame shape:", KP_df.shape)
    # convert pos_class to categories
    #c = KP_df['pos_class'].astype('category')
    #print(c.cat.categories, c.cat.codes)

    KP_df['pos_class'] = KP_df['pos_class'].astype('category').cat.codes
    # skipping (0-3) colomns , return values of all rows and columns from 4 to last
    features = KP_df.iloc[:,6:].values
    #return values of pose_class 
    pose_class = KP_df['pos_class'].values
    # normalize keypoints 
    print('features shape', len(features[0]))
    features = normalize_min_(features)
    # append multiple rows to create a sequence of data
    if n_frames>1:
        for i in range(features.shape[0]-n_frames):
            if pose_class[i]==pose_class[i+n_frames]:
                data.append(features[i:i+n_frames,...])
                label_sequence = pose_class[i:i+n_frames]
                #with open('label.txt',"a") as file:
                #file.write(str(label_sequence)+"\n")
                unique, counts = np.unique(label_sequence, return_counts=True)
                label.append(unique[np.argmax(counts)])
    elif n_frames==1:
        print('creating single frame')
        for i in range(features.shape[0]):
            data.append(features[i])
            label.append(pose_class[i])

    data , label =  np.array(data, dtype = np.float), np.array(label, dtype = np.int_)
    return data , label

# min-max normalization to scale the x, y coordinates in range (0-1) 
def normalize_min_(pose:np.ndarray):
    pose = pose.reshape(len(pose),-1,2)
    for i in range(len(pose)):
        xmin = np.min(pose[i,:,0]) 
        ymin = np.min(pose[i,:,1])
        xlen = np.max(pose[i,:,0]) - xmin
        ylen = np.max(pose[i,:,1]) - ymin

        if(xlen==0): pose[i,:,0]=0
        else:
            pose[i,:,0] -= xmin 
            pose[i,:,0] /= xlen

        if(ylen==0): pose[i,:,1]=0
        else:
            pose[i,:,1] -= ymin
            pose[i,:,1] /= ylen
    return pose