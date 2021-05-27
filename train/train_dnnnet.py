import os 
import numpy as np
import time
import copy
from collections import defaultdict

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from fallModels.models import Net
from dataloader import SinglePose2dDataset, SinglePose3dDataset

currTime = time.asctime( time.localtime(time.time()))[4:-5]
currTime = currTime.split(' ')
# currTime = currTime[0]+'_'+currTime[1]+'_'+currTime[2]
currTime = currTime[0]+'_'+currTime[1]

class trainDNN():
    def __init__(self):
        print('Initializing training...')

    #training function
    @staticmethod
    def train(model, dataloader, dataset_size, dataloader3d, dataset3d_size, num_epochs=3000):
    
        history = defaultdict(list)
        # define the optimization
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5, factor=0.1,verbose=True)
        
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        loss_ = 100
        conf_train = 0.0
        conf_valid = 0.0
        # enumerate over epochs
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs-1))
            print('-'*10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train() # St model to training mode
                
                else:
                    model.eval() # Set model to evaluate mode
                    
                running_loss = 0.0
                running_corrects = 0.0
                # Initialize the prediction and label lists(tensors)
                pred_tensor  = torch.zeros(0,dtype=torch.long, device='cpu')
                class_tensor = torch.zeros(0,dtype=torch.long, device='cpu')
                
                # iterate over two dataloaders simultaneously
                dataloader_iterator = iter(dataloader[phase])
                for i, (inputs2, targets2)  in enumerate(dataloader3d[phase]):


                    inputs2  = inputs2.to(device)
                    targets2 = targets2.to(device)
                    try:
                        inputs1, targets1 = next(dataloader_iterator)
                    except StopIteration:
                        dataloader_iterator = iter(dataloader[phase])
                        inputs1, targets1 = next(dataloader_iterator)
                    
                    inputs1  = inputs1.to(device)
                    targets1 = targets1.to(device)

                    # clear the parameter gradients
                    optimizer.zero_grad()
                    
                    # forward
                    # track history if only in train pahse
                    with torch.set_grad_enabled(phase=='train'):
                        
                        # compute model outputs
                        #print(targets1, targets2)
                        raw_preds1,raw_preds2,raw_preds3,class_probs = model(inputs1, inputs2)
                        
                        # calculate outputs
                        _, preds = torch.max(class_probs, dim=1)
                        
                        # calculate the loss
                        loss1 = criterion(raw_preds1, targets1)
                        loss2 = criterion(raw_preds2, targets2)
                        loss3 = criterion(raw_preds3, targets2)
                        loss = loss1+loss2
                        
                        # backward + optimize only ig in training phase
                        if phase == 'train':
                            # calculate gradient
                            loss1.backward(retain_graph=True)
                            loss2.backward(retain_graph=True)
                            loss3.backward(retain_graph=True)
                            # update model weights
                            optimizer.step()
                    
                    #statistics
                    running_loss += loss.item() * inputs2.size(0)
                    running_corrects += torch.sum(preds==targets2)
                    
                    # Append batch prediction results
                    pred_tensor = torch.cat([pred_tensor,preds.view(-1).cpu()])
                    class_tensor  = torch.cat([class_tensor,targets2.view(-1).cpu()])
                    
                
                #epoch loss and accuracy
                epoch_loss = running_loss / dataset3d_size[phase]
                epoch_acc  = running_corrects.double() / dataset3d_size[phase]
                history[phase].append((epoch_loss, epoch_acc, ))
                
                if phase=='valid':
                    scheduler.step()

                # Confusion matrix
                conf_mat = confusion_matrix(class_tensor.numpy(), pred_tensor.numpy())
                # Per-class accuracy
                #per_class_accuracy = np.round(100*conf_mat.diagonal()/conf_mat.sum(1),4)
                #Precision, Recall, F1_Score
                precision = precision_score(class_tensor.numpy(), pred_tensor.numpy(), average=None)
                recall = recall_score(class_tensor.numpy(), pred_tensor.numpy(), average=None)
                f_score = f1_score(class_tensor.numpy(), pred_tensor.numpy(), average=None)
                
                print('{} : Loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                #print('{} : Confusion Matrix: {}'.format(phase, conf_mat))
                #print('{} : Precision per class: {}'.format(phase, np.round(precision,4)))
                #print('{} : Recall per class: {}'.format(phase, np.round(recall,4)))
                print('{} : F1_Score per class: {}'.format(phase, np.round(f_score,4)))
                print()
                
                if phase== 'valid' and loss_>epoch_loss:
                    epoch_ = epoch
                    loss_  = epoch_loss
                    conf_valid = conf_mat
                    #best_model_wts = copy.deepcopy(model.state_dict())
                    trainDNN.save_model(model, optimizer, loss_, epoch_acc, epoch_, save_path=r'checkpoints/dnn2d3d_'+currTime)
                
                if phase== 'train' and epoch_acc>best_acc:
                    conf_train = conf_mat
                    
            print()
        
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed //60, time_elapsed %60))
        print('Best val Acc: {:4f}'.format(epoch_acc))
        
        return history, model, conf_train, conf_valid
    @staticmethod
    def save_model(model, optimizer, loss, acc, epoch, save_path):
    
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print('SAVING EPOCH %d'%epoch)
        filename = 'epoch_%d'%epoch + '_loss_%f.pth'%loss
        SAVE_FILE = os.path.join(save_path,filename)
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'acc':  acc,
                }, SAVE_FILE)


    

if __name__ == '__main__':

    DNN_model   = Net(class_num=2, input2d=34, input3d=51, ).to(device)
    #get test dataloaders
    dataloader, dataset_size = SinglePose2dDataset.get2dData(reshape=False, bs=16,n_frames=1)
    dataloader3d, dataset3d_size = SinglePose3dDataset.get3dData(reshape=False, bs=16,n_frames=1)
    print(dir(dataloader['train']), dataset_size)
    print(dir(dataloader3d['train']), dataset3d_size)
    #train the model
    history, model, conf_train, conf_valid = trainDNN.train(DNN_model, dataloader, dataset_size, 
    dataloader3d, dataset3d_size, num_epochs=500)
    #plot the model statistics 
    #plot_Statistics(history,conf_train, conf_valid,name='dnntiny')