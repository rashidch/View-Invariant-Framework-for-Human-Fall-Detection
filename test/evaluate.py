
import os
import numpy as np
import time
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler 


from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn import model_selection
from train.dataLoader import TwoDimensionaldataset, get2dData
from train.plot_statics import plot_Statistics
from fallModels.fallModelsModule import getModel
from test.classifier_config.apis import getFallModelcfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(prog="Fall Detection App",description="This program reads video frames and predicts fall",
                                epilog="Enjoy the program")
parser.add_argument("--checkpoint",type=str,required=False,default="source/pretrained_models/fast_res50_256x192.pth",
                    help="checkpoint file name",)
parser.add_argument("-cls","--classifier", dest="classmodel", type=str, default="dstanet", 
                    help="choose classifer model, defualt dnn model")
args = parser.parse_args()

def load_model(cfg, numFrames, pose2dSize):
    model = getModel(cfg.MODEL,cfg.tagI2W, n_frames=numFrames,pose2d_size=pose2dSize, pose3d=51)
    print('CheckPoint:',cfg.CHEKPT)
    ckpt  = torch.load(cfg.CHEKPT, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def evalClassifier(stmodel,dataloader, dataset_size, num_channel, num_point, num_frames):
    
    #history = defaultdict(list)
    phase = 'valid'
    print("-" * 10)
    stmodel.eval()  # Set model to evaluate mode
    running_corrects = 0.0
    # Initialize the prediction and label lists(tensors)
    pred_tensor = torch.zeros(0, dtype=torch.long, device="cpu")
    class_tensor = torch.zeros(0, dtype=torch.long, device="cpu")

    # enumerate over mini_batch
    for _, (inputs, targets) in enumerate(dataloader[phase]):
        #print(inputs.shape)
        #[16,32,17,2]

        inputs = inputs.permute(0,3,1,2).contiguous()
        inputs = inputs.to(device)
        targets = targets.to(device)
        # forward
        with torch.no_grad():
            # compute model outputs
            raw_preds, class_probs = stmodel(inputs)
            # calculate outputs
            _, preds = torch.max(class_probs, dim=1)
        # statistics
        running_corrects += torch.sum(preds == targets)

        # Append batch prediction results
        pred_tensor  = torch.cat([pred_tensor, preds.view(-1).cpu()])
        class_tensor = torch.cat([class_tensor, targets.view(-1).cpu()])

        # epoch loss and accuracy
        epoch_acc = running_corrects.double() / dataset_size[phase]
        # Confusion matrix
        conf_mat = confusion_matrix(class_tensor.numpy(), pred_tensor.numpy())
        # Precision, Recall, F1_Score
        precision = precision_score(class_tensor.numpy(), pred_tensor.numpy(), average=None)
        recall = recall_score(class_tensor.numpy(), pred_tensor.numpy(), average=None)
        f_score = f1_score(class_tensor.numpy(), pred_tensor.numpy(), average=None)
        print("{} : Acc: {:.4f}".format(phase,epoch_acc))
        print('{} : Confusion Matrix: {}'.format(phase, conf_mat))
        print('{} : Precision per class: {}'.format(phase, np.round(precision,4)))
        print('{} : Recall per class: {}'.format(phase, np.round(recall,4)))
        print("{} : F1_Score per class: {}".format(phase, np.round(f_score, 4)))
        print()
        print()
 
def eval():
    num_point=11
    num_frames=10
    batchSize=1000
    num_channel=2

    cfg   = getFallModelcfg(args)
    stmodel = load_model(cfg, numFrames=num_frames,pose2dSize=22)
    # get test dataloaders
    dataloader, dataset_size = get2dData(bs=batchSize, n_frames=num_frames,reshape=False)
    evalClassifier(stmodel,dataloader, dataset_size, num_channel, num_point, num_frames)
    #plot the model statistics 
    #plot_Statistics(history,conf_train, conf_valid,name='stnet',epochs=num_epochs)
    

if __name__=='__main__':
    eval()