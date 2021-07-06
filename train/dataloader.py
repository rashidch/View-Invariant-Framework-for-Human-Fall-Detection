import pandas as pd
import os 
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import SubsetRandomSampler 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.model_selection import train_test_split

#create single pose 2d dataset for dnn model
class SinglePose2dDataset(Dataset):
    
    def __init__(self, n_frames=5):
        
        # get data and label from csvget 
        dataset, labels = SinglePose2dDataset.ReadPose2dData(n_frames)
        
        print('Dataset',dataset.shape, 'labels',labels.shape)
        
        self.X = dataset
        self.y = labels
    
    # read dataset from csv file
    @staticmethod
    def ReadPose2dData(n_frames):
        
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
        SinglePose2dDataset.normalize_min_(features)
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
    @staticmethod
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
    
    # number of rows in the dataset
    def __len__(self):
        
        return len(self.X)
        
    # get a row at an index
    def __getitem__(self, idx):
        
        data  = torch.tensor(self.X[idx], dtype=torch.float) 
        label = torch.tensor(self.y[idx], dtype=torch.long)
        
        return [data,label]
    
    # get indexes for train and test rows
    
    '''
    def get_splits(self, n_test = 0.2, n_valid=0.2):
        
        # determine sizes 
        test_size = round(n_test * len(self.X))
        valid_size = round(n_valid * len(self.X))
        train_size = len(self.X)-(test_size+valid_size)
        print(train_size, valid_size, test_size)
        # calculate the split 
        return random_split(self, [train_size, valid_size, test_size])
    '''
    
    def get_class_labels(self):
        
        labels = ["Fall","Stand"]
        
        return labels
    
    def reshape_features(self):
        self.X = self.X.reshape(-1, self.X.shape[1]*self.X.shape[2])

    #prepare pytorch data loaders 
    @staticmethod
    def get2dData(reshape=False, n_frames=1, bs=32, n_test = 0.35):
    
        # load pose dataset
        dataset = SinglePose2dDataset(n_frames=n_frames)
        targets = dataset.y
        
        # reshape from N,10,34 to N,Num_Features
        if reshape:
            dataset.reshape_features()
        
        #stratified train test split
        train_idx, valid_idx = train_test_split(np.arange(len(targets)), test_size=n_test, stratify=targets)
        
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_dl  = DataLoader(dataset, batch_size=bs, num_workers=4, sampler=train_sampler, drop_last=True)
        valid_dl  = DataLoader(dataset, batch_size=bs, sampler=valid_sampler, drop_last=True)
        
        return {'train':train_dl, 'valid':valid_dl}, {'train':len(train_sampler), 'valid':len(valid_sampler)}

#create single pose 2d dataset for dnn model
class SinglePose3dDataset(Dataset):
    
    
    def __init__(self, n_frames=5):
        
        # get data and label from csvget 
        dataset, labels = SinglePose3dDataset.ReadPose3dData(n_frames)
        
        print('Dataset',dataset.shape, 'labels',labels.shape)
        
        self.X = dataset
        self.y = labels
    
    # read dataset from csv file
    @staticmethod
    def ReadPose3dData(n_frames):
        
        # get csv file path
        curr_dir = os.getcwd()
        csv_file_path = os.path.join(curr_dir, 'dataset/DataCSV/cam7_3D_Original.csv')
        
        # list for storing data and labels
        data  = []
        label = []
        
        # lenth of sequence
        #n_frames = SinglePoseDataset.n_frames
        
        # read csv file
        KP_df = pd.read_csv(csv_file_path)
        #print("DataFrame shape:", KP_df.shape)
        # convert pos_class to categories
        KP_df['pos_class'] = KP_df['pos_class'].astype('category')
        KP_df['pos_class'] = KP_df['pos_class'].cat.codes

        # skipping (0-3) colomns , return values of all rows and columns from 4 to last
        features = KP_df.iloc[:,6:].values
        #return values of pose_class 
        pose_class = KP_df['pos_class'].values
        # normalize keypoints 
        print('features shape', len(features[0]))
        SinglePose3dDataset.normalize_min_(features)
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
    @staticmethod
    def normalize_min_(pose:np.ndarray):
        print(pose.shape)
        pose = pose.reshape(len(pose),-1,3)
        for i in range(len(pose)):
            xmin = np.min(pose[i,:,0]) 
            ymin = np.min(pose[i,:,1])
            zmin = np.min(pose[i,:,2])
            xlen = np.max(pose[i,:,0]) - xmin
            ylen = np.max(pose[i,:,1]) - ymin
            zlen = np.max(pose[i,:,2]) - zmin

            if(xlen==0): pose[i,:,0]=0
            else:
                pose[i,:,0] -= xmin 
                pose[i,:,0] /= xlen

            if(ylen==0): pose[i,:,1]=0
            else:
                pose[i,:,1] -= ymin
                pose[i,:,1] /= ylen

            if(zlen==0): pose[i,:,2]=0
            else:
                pose[i,:,2] -= zmin
                pose[i,:,2] /= zlen
        return pose
    
    # number of rows in the dataset
    def __len__(self):
        
        return len(self.X)
        
    # get a row at an index
    def __getitem__(self, idx):
        
        data  = torch.tensor(self.X[idx], dtype=torch.float) 
        label = torch.tensor(self.y[idx], dtype=torch.long)
        
        return [data,label]
    
    # get indexes for train and test rows
    
    '''
    def get_splits(self, n_test = 0.2, n_valid=0.2):
        
        # determine sizes 
        test_size = round(n_test * len(self.X))
        valid_size = round(n_valid * len(self.X))
        train_size = len(self.X)-(test_size+valid_size)
        print(train_size, valid_size, test_size)
        # calculate the split 
        return random_split(self, [train_size, valid_size, test_size])
    '''
    
    def get_class_labels(self):
        
        labels = ["Fall","Stand"]
        
        return labels
    
    def reshape_features(self):
        self.X = self.X.reshape(-1, self.X.shape[1]*self.X.shape[2])

    #prepare pytorch data loaders 
    @staticmethod
    def get3dData(reshape=False, n_frames=1, bs=32, n_test = 0.35):
    
        # load pose dataset
        dataset = SinglePose3dDataset(n_frames=n_frames)
        targets = dataset.y
        
        # reshape from N,10,34 to N,Num_Features
        if reshape:
            dataset.reshape_features()
        
        #stratified train test split
        train_idx, valid_idx = train_test_split(np.arange(len(targets)), test_size=n_test, stratify=targets)
        
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_dl  = DataLoader(dataset, batch_size=bs, num_workers=4, sampler=train_sampler, drop_last=True)
        valid_dl  = DataLoader(dataset, batch_size=bs, sampler=valid_sampler, drop_last=True)
        
        return {'train':train_dl, 'valid':valid_dl}, {'train':len(train_sampler), 'valid':len(valid_sampler)}