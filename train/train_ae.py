import os 
import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from fallModels.models import FallNet, GenNet
from dataloader import prepare_data

import argparse
parser = argparse.ArgumentParser(description="Compress Num")
parser.add_argument('--Num',type=int,default=15)
args = parser.parse_args()
Num = args.Num
Seq_len = 30
dir_file = 'checkpoints_fallmodel/aelstm/with_sigmoid/cpr{}_Fall1-Seq_{}'.format(Num,Seq_len)
if not(os.path.exists(dir_file)):
   os.mkdir(dir_file)
   
#training function
def train_model(model_ft, model_gen, dataloaders, dataset_sizes, num_epochs=100):
    
    history = defaultdict(list)
    # define the optimization
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    MSEdis = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(list(model_ft.parameters())+list(model_gen.parameters()),lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5, factor=0.1,verbose=True)
    
    since = time.time()
    best_acc = 0.0
    conf_train = 0.0
    conf_valid = 0.0
    count = 0
    lamda = 10  # regularization parameter
    # enumerate over epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('*'*10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model_ft.train() # Set model_ft to training mode
                model_gen.train() # Set model_gen to training mode
            
            else:
                model_ft.eval() # Set model to evaluate mode
            
            count = 0   
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
                
                    Deinputs  = model_gen(inputs)
                    De_preds, De_probs  = model_ft(Deinputs)
                    Out_preds, Out_probs = model_ft(inputs)
                
                    # calculate the loss
                    loss_De_preds = criterion(De_preds,targets)
                    loss_Out_preds = criterion(Out_preds,targets)
                    loss_mse = MSEdis(Deinputs, inputs)*lamda

                    loss  = loss_De_preds + loss_Out_preds + loss_mse
                    #loss  = loss_De_preds + loss_mse
                    
                    # backward + optimize only ig in training phase
                    if phase == 'train':
                        # calculate gradient
                        loss.backward()
                    
                        # update model weights
                        optimizer.step()

                        count +=1
                        if count%100==0:
                            print('Epoch:{}:loss_De_preds:{:.3f} loss_out_preds:{:.3f} loss_mse:{:.3f}'.format(
                                epoch,loss_De_preds.item(),loss_Out_preds.item(),loss_mse.item()))

                    # calculate outputs
                    _, preds = torch.max(De_probs, dim=1)
                
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
            #Precision, Recall, F1_Score
            precision = precision_score(class_tensor.numpy(), pred_tensor.numpy(), average=None)
            recall = recall_score(class_tensor.numpy(), pred_tensor.numpy(), average=None)
            f_score = f1_score(class_tensor.numpy(), pred_tensor.numpy(), average=None)
            
            print('{} : Loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} : F1_Score per class: {}'.format(phase, np.round(f_score,4)))
            print()
            
            if phase== 'valid' and (epoch+1)%100==0:
                epoch_ = epoch
                loss_  = epoch_loss
                conf_valid = conf_mat
                model_out_path = dir_file + "/classSig_epoch_{}.pth".format(epoch)
                torch.save(model_ft, model_out_path)
                model_out_path = dir_file + "/gen_epoch_{}.pth".format(epoch)
                torch.save(model_gen, model_out_path)
            
            if phase== 'train' and epoch_acc>best_acc:
                conf_train = conf_mat
                   
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed //60, time_elapsed %60))
    print('Best val Acc: {:4f}'.format(epoch_acc))
    
    return history, model_ft, conf_train, conf_valid

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

    model_ft = FallNet(input_dim=24, class_num=2).to(device)
    model_gen = GenNet(Num).to(device)
    #total_params = sum(p.numel() for p in LSTM_model.parameters() if p.requires_grad)
    #print('Total Model Parameters:',total_params)

    #get test dataloaders
    dataloaders, dataset_sizes = prepare_data(bs=32, seq_len=30)
    print("dataset size", dataset_sizes)
    #train the model
    history, model, conf_train, conf_valid = train_model(model_ft,model_gen, dataloaders, 
                                                            dataset_sizes,num_epochs=300)
    #plot the model statistics 
    #plot_Statistics(history,conf_train, conf_valid)