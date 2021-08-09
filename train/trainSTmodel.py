import os
import numpy as np
import time
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn import model_selection
from fallModels.models import dnntiny, dnnnet
from fallModels.stmodel import DSTANet 
#from fallModels.STAN_AE import DSTANet, GenNet 
from train.dataLoader import TwoDimensionaldataset, get2dData
from train.plot_statics import plot_Statistics
#from train.focalLoss import FocalLoss
from train.optimizer_choose import optimizer_choose, optimizer_choose2

currTime = time.asctime(time.localtime(time.time()))[4:-5]
currTime = currTime.split(' ')
currTime = currTime[0]+'_'+currTime[2]+'_'+currTime[3]


def train(stmodel,dataloader, dataset_size, num_channel, num_point, n_frames, num_epochs, fold=None):
    
    #print(dataset_size['train'],dataset_size['valid'])
    history = defaultdict(list)
    # define the optimization
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    MSEdis = torch.nn.MSELoss().to(device)
    optimizer = optimizer_choose(stmodel, optimizer='sgd_nev')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',threshold=1e-4,
                                #cooldown=0,patience=5,threshold_mode='abs', factor=0.1,verbose=True)


    since = time.time()
    best_acc = 0.0
    loss_ = 100
    conf_train = 0.0
    conf_valid = 0.0
    lamda = 1  # regularization parameter
    # enumerate over epochs
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "valid"]:
            if phase == "train":
                stmodel.train()  # St model to training mode

            else:
                stmodel.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Initialize the prediction and label lists(tensors)
            #Depred_tensor = torch.zeros(0, dtype=torch.long, device="cpu")
            pred_tensor = torch.zeros(0, dtype=torch.long, device="cpu")
            class_tensor = torch.zeros(0, dtype=torch.long, device="cpu")

            # enumerate over mini_batch
            for _, (inputs, targets) in enumerate(dataloader[phase]):
                #print(inputs.shape)
                #[16,32,17,2]

                inputs = inputs.permute(0,3,1,2).contiguous()
                #inputs = inputs.reshape(-1,n_frames,num_point*num_channel)
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # clear the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train pahse
                with torch.set_grad_enabled(phase == "train"):
                    #stinputs = inputs.reshape(-1,num_frame,num_point,num_channel).permute(0,3,1,2).contiguous()
                    # compute model outputs
                    raw_preds, class_probs = stmodel(inputs)
                    # calculate outputs
                    _, preds = torch.max(class_probs, dim=1)
                    # calculate the loss
                   
                    loss = criterion(raw_preds, targets)
                    # backward + optimize only ig in training phase
                    if phase == "train":
                        # calculate gradient
                        loss.backward()

                        # update model weights
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == targets)

                # Append batch prediction results
                pred_tensor  = torch.cat([pred_tensor, preds.view(-1).cpu()])
                class_tensor = torch.cat([class_tensor, targets.view(-1).cpu()])

            # epoch loss and accuracy
            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]
            history[phase].append(
                (
                    epoch_loss,
                    epoch_acc,
                )
            )

            if phase == "valid":
                #scheduler.step(epoch_loss)
                scheduler.step()

            # Confusion matrix
            conf_mat = confusion_matrix(class_tensor.numpy(), pred_tensor.numpy())
            # Per-class accuracy
            # per_class_accuracy = np.round(100*conf_mat.diagonal()/conf_mat.sum(1),4)
            # Precision, Recall, F1_Score
            precision = precision_score(class_tensor.numpy(), pred_tensor.numpy(), average=None)
            recall = recall_score(class_tensor.numpy(), pred_tensor.numpy(), average=None)
            f_score = f1_score(class_tensor.numpy(), pred_tensor.numpy(), average=None)


            print("{} : Loss: {:.4f}, Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            print('{} : Confusion Matrix: {}'.format(phase, conf_mat))
            print('{} : Precision per class: {}'.format(phase, np.round(precision,4)))
            print('{} : Recall per class: {}'.format(phase, np.round(recall,4)))
            print("{} : F1_Score per class: {}".format(phase, np.round(f_score, 4)))
            print()


            if phase == "valid" and loss_ > epoch_loss:
                epoch_ = epoch
                loss_ = epoch_loss
                conf_valid = conf_mat
                saveModel(
                    stmodel, optimizer, loss_, epoch_acc, epoch_,fold,save_path=r"checkpoints/STANnet_"+ str(currTime)
                )

            if phase == "train" and epoch_acc > best_acc:
                conf_train = conf_mat

        print()

    saveModel(
    stmodel, optimizer, epoch_loss, epoch_acc, epoch,fold,save_path=r"checkpoints/STANnet_"+str(currTime))

    
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(epoch_acc))

    return history, _, conf_train, conf_valid

    
def saveModel(model, optimizer, loss, acc, epoch,fold, save_path): 

    if not os.path.exists(save_path):
        print(save_path)
        os.makedirs(save_path)
    print("SAVING EPOCH %d" % epoch)
    if fold!=None:
        filename = "fold_%d" % fold+"_epoch_%d" % epoch + "_loss_%f.pth" % loss
    else:
        filename = "epoch_%d" % epoch + "_loss_%f.pth" % loss
    SAVE_FILE = os.path.join(save_path, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "acc": acc,
        },
        SAVE_FILE,)


def corssValiation(DNN_model,kfolds=5,n_frames=1, bs=32, num_epochs=1500):
    
    cross_valid = model_selection.StratifiedKFold(n_splits=kfolds, shuffle=False)

    #get pose dataset
    dataset = TwoDimensionaldataset(n_frames=n_frames)
    for fold, (train_idx, valid_idx) in enumerate(cross_valid.split(X=dataset.X, y=dataset.y)):
        print("------------Corss Validation KFold {}--------".format(fold))
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_dl  = DataLoader(dataset, batch_size=bs, num_workers=4, sampler=train_sampler, drop_last=True)
        valid_dl  = DataLoader(dataset, batch_size=bs, sampler=valid_sampler, drop_last=True)

        dataloader  = {'train':train_dl, 'valid':valid_dl} 
        dataset_size = {'train':len(train_sampler), 'valid':len(valid_sampler)}
        # train the model
        history, _, conf_train, conf_valid = train(DNN_model, dataloader, dataset_size, num_epochs=num_epochs, fold=fold)
    #plot the model statistics 
    plot_Statistics(history,conf_train, conf_valid,name='stnet',epochs=num_epochs)
    

def trainvalSplit(stnet,num_channel, num_point,n_frames,bs, num_epochs):

    # get test dataloaders
    dataLoader, datasetSize = get2dData(bs=bs, n_frames=n_frames,reshape=False)
    # dataloader3d, dataset3d_sizes = SinglePose3dDataset.get3dData(reshape=False, bs=16,n_frames=1)
    # train the model
    history, _, conf_train, conf_valid = train(stnet,dataLoader, datasetSize, num_channel,
                                                num_point, n_frames, num_epochs)
    #plot the model statistics 
    plot_Statistics(history,conf_train, conf_valid,name='stnet',epochs=num_epochs)


if __name__ == "__main__":

    # set training mode
    training_mode = "train_test_split"

    '''
    config = [[64, 64, 16, 1], [64, 64, 16, 1],
              [64, 128, 32, 2], [128, 128, 32, 1],
              [128, 256, 64, 2], [256, 256, 64, 1],
              [256, 256, 64, 1], [256, 256, 64, 1],
              ]
    

    
    config = [[64, 64, 16, 1], [64, 64, 16, 1],
              [64, 128, 32, 2], [128, 128, 32, 1]]
    '''
    config = [[64, 64, 16, 1], [64, 64, 16, 1]]
    
    num_class=2
    num_point=11
    num_frame=10
    batchSize=16
    num_epochs = 120
    Num = 25
    num_channel=2
    stnet = DSTANet(num_class=num_class, num_point=num_point, num_frame=num_frame,num_channel=2,config=config).to(device)  # .cuda()
   
    
    if training_mode == "train_test_split":
        trainvalSplit(stnet,num_channel=num_channel, num_point=num_point,n_frames=num_frame, bs=batchSize, num_epochs=num_epochs)
        
    elif training_mode == "cross_validation":
       corssValiation(stnet,kfolds=2,n_frames=num_frame, bs=batchSize, num_epochs=num_epochs)