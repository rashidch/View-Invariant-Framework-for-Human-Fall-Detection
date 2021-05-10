import os 
import numpy as np
import time
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from fallModels.models import FallModel, DNN_tiny
from dataloader import prepare_data

def save_model(model, optimizer, loss, acc, epoch, save_path):
    
    #base_dir = os.path.basename(os.getcwd())
    
    #if base_dir =='train':
    #parent_dir = os.path.dirname(os.getcwd())
    #os.chdir(parent_dir)
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

#training function
def train_model(model, dataloaders, dataset_sizes, num_epochs=3000):
    
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
            
            # enumerate over mini_batch
            for i, (inputs, targets)  in enumerate(dataloaders[phase]):
                #print('shape of input', inputs.shape)
                inputs  = inputs.to(device)
                targets = targets.to(device)
                
                # clear the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train pahse
                with torch.set_grad_enabled(phase=='train'):
                    
                    # compute model outputs
                    raw_preds , class_probs = model(inputs)
                    
                    # calculate outputs
                    _, preds = torch.max(class_probs, dim=1)
                    
                    # calculate the loss
                    loss = criterion(raw_preds, targets)
                    
                    # backward + optimize only ig in training phase
                    if phase == 'train':
                        # calculate gradient
                        loss.backward()
                    
                        # update model weights
                        optimizer.step()
                
                #statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds==targets)
                
                # Append batch prediction results
                pred_tensor = torch.cat([pred_tensor,preds.view(-1).cpu()])
                class_tensor  = torch.cat([class_tensor,targets.view(-1).cpu()])
                
            
            #epoch loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]
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
                save_model(model, optimizer, loss_, epoch_acc, epoch_, save_path=r'checkpoints_fallmodel/act_dnntiny_1')
            
            if phase== 'train' and epoch_acc>best_acc:
                conf_train = conf_mat
                   
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed //60, time_elapsed %60))
    print('Best val Acc: {:4f}'.format(epoch_acc))
    
    return history, model, conf_train, conf_valid

def plot_Statistics(history, conf_train, conf_valid):

    train_acc  = []
    train_loss = []
    val_acc    = []
    val_loss   = []
    for train_item, val_item in zip(history['train'],history['valid']):
        
        val_loss.append(val_item.__getitem__(0))
        val_acc.append(val_item.__getitem__(1).cpu().detach().numpy())
        
        train_loss.append(train_item.__getitem__(0))
        train_acc.append(train_item.__getitem__(1).cpu().detach().numpy())
    
    params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)


    num_epochs=range(1000)
    plt.figure(figsize=(12, 6))
    plt.plot(num_epochs, train_acc, label='Training Accuracy')
    plt.plot(num_epochs, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy', fontsize=18)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('plots/dnntiny_seq5_acc.jpg')
    plt.show()

    num_epochs=range(1000)
    plt.figure(figsize=(12, 6))
    plt.plot(num_epochs, train_loss, label='Training Loss')
    plt.plot(num_epochs, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('plots/dnntiny_seq5_loss.jpg')
    plt.show()

    #set the size of figure 542129345
    plt.figure(figsize=(8,8))
    #normalize each column (class) with total datapoints in that column  
    conf_train = conf_train.astype('float')/conf_train.sum(axis=1)*100
    #plot confusion matrix 
    p=sns.heatmap(conf_train, xticklabels=['Fall','Stand','Tie'], yticklabels=['Fall','Stand','Tie'],
                cbar=False, annot=True, cmap='coolwarm',robust=True, fmt='.1f',annot_kws={'size':20})
    plt.title('Training matrix: Actual labels Vs Predicted labels')
    plt.savefig('plots/dnntiny_seq5_train_cf.png')

    #set the size of figure 
    plt.figure(figsize=(8,8))
    #normalize each column (class) with total datapoints in that column  
    conf_valid = conf_valid.astype('float')/conf_valid.sum(axis=1)*100
    #plot confusion matrix 
    p=sns.heatmap(conf_valid, xticklabels=['Fall','Stand','Tie'], yticklabels=['Fall','Stand','Tie'],
                cbar=False, annot=True, cmap='coolwarm',robust=True, fmt='.1f',annot_kws={'size':20})
    plt.title('Validation matrix: Actual labels vs Predicted labels')
    plt.savefig('plots/dnntiny_seq5_valid_cf.png')

if __name__ == '__main__':

    DNN_model   = DNN_tiny(input_dim=24, class_num=2).to(device)
    #total_params = sum(p.numel() for p in LSTM_model.parameters() if p.requires_grad)
    #print('Total Model Parameters:',total_params)

    #get test dataloaders
    dataloaders, dataset_sizes = prepare_data(bs=32, seq_len=1)
    print("dataset size", dataset_sizes)
    #train the model
    history, model, conf_train, conf_valid = train_model(DNN_model, dataloaders, dataset_sizes,num_epochs=500)
    #plot the model statistics 
    #plot_Statistics(history,conf_train, conf_valid)