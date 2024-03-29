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
from train.dataloader import SinglePose2dDataset
from train.plot_statics import plot_Statistics

currTime = time.asctime(time.localtime(time.time()))[4:-5]
currTime = currTime.split(' ')
currTime = currTime[0]+'_'+currTime[1]+currTime[2]


class trainDNN:
    def __init__(self):
        print("Initializing training...")

    # training function
    @staticmethod
    def train(model, dataloader, dataset_size, num_epochs=3000, fold=None):

        history = defaultdict(list)
        # define the optimization
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=100, factor=0.7,verbose=True)

        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        loss_ = 100
        conf_train = 0.0
        conf_valid = 0.0
        # enumerate over epochs
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "valid"]:
                if phase == "train":
                    model.train()  # St model to training mode

                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0.0
                # Initialize the prediction and label lists(tensors)
                pred_tensor = torch.zeros(0, dtype=torch.long, device="cpu")
                class_tensor = torch.zeros(0, dtype=torch.long, device="cpu")

                # enumerate over mini_batch
                for i, (inputs, targets) in enumerate(dataloader[phase]):
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # clear the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train pahse
                    with torch.set_grad_enabled(phase == "train"):

                        # compute model outputs
                        raw_preds, class_probs = model(inputs)

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
                    pred_tensor = torch.cat([pred_tensor, preds.view(-1).cpu()])
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
                # print('{} : Confusion Matrix: {}'.format(phase, conf_mat))
                print('{} : Precision per class: {}'.format(phase, np.round(precision,4)))
                print('{} : Recall per class: {}'.format(phase, np.round(recall,4)))
                print("{} : F1_Score per class: {}".format(phase, np.round(f_score, 4)))
                print()

                if phase == "valid" and loss_ > epoch_loss:
                    epoch_ = epoch
                    loss_ = epoch_loss
                    conf_valid = conf_mat
                    # best_model_wts = copy.deepcopy(model.state_dict())
                    trainDNN.save_model(
                        model, optimizer, loss_, epoch_acc, epoch_,fold,save_path=r"checkpoints/alph_dnnnet_" + currTime
                    )

                if phase == "train" and epoch_acc > best_acc:
                    conf_train = conf_mat

            print()

        trainDNN.save_model(
        model, optimizer, loss_, epoch_acc, epoch_,fold,save_path=r"checkpoints/alph_dnnnet_" + currTime
                    )
        time_elapsed = time.time() - since
        print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        print("Best val Acc: {:4f}".format(epoch_acc))

        return history, model, conf_train, conf_valid

    @staticmethod
    def save_model(model, optimizer, loss, acc, epoch,fold, save_path): 

        if not os.path.exists(save_path):
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
            SAVE_FILE,
        )

    @staticmethod
    def trainWithKfolds(DNN_model,kfolds=5,n_frames=1, bs=32, num_epochs=1500):
        
        cross_valid = model_selection.StratifiedKFold(n_splits=kfolds, shuffle=False)

        #get pose dataset
        dataset = SinglePose2dDataset(n_frames=n_frames)
        for fold, (train_idx, valid_idx) in enumerate(cross_valid.split(X=dataset.X, y=dataset.y)):
            print("------------Corss Validation KFold {}--------".format(fold))
            #print(len(train_idx), len(valid_idx))
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            train_dl  = DataLoader(dataset, batch_size=bs, num_workers=4, sampler=train_sampler, drop_last=True)
            valid_dl  = DataLoader(dataset, batch_size=bs, sampler=valid_sampler, drop_last=True)

            dataloader  = {'train':train_dl, 'valid':valid_dl} 
            dataset_size = {'train':len(train_sampler), 'valid':len(valid_sampler)}
            # train the model
            history, _, conf_train, conf_valid = trainDNN.train(DNN_model, dataloader, dataset_size, num_epochs=num_epochs, fold=fold)
        #plot the model statistics 
        plot_Statistics(history,conf_train, conf_valid,name='dnn2d',epochs=num_epochs)
    
    @staticmethod
    def trainWithsplit(DNN_model,n_frames=1, bs=32, num_epochs=1500):

        # get test dataloaders
        dataloaders, dataset_sizes = SinglePose2dDataset.get2dData(reshape=False, bs=bs, n_frames=n_frames)
        # dataloader3d, dataset3d_sizes = SinglePose3dDataset.get3dData(reshape=False, bs=16,n_frames=1)
        print(dataloaders, dataset_sizes)
        # train the model
        history, model, conf_train, conf_valid = trainDNN.train(DNN_model, dataloaders, dataset_sizes, num_epochs=num_epochs)
        #plot the model statistics 
        plot_Statistics(history,conf_train, conf_valid,name='dnnnet2d',epochs=num_epochs)


if __name__ == "__main__":

    # set training mode
    training_mode = "train_test_split"

    #DNN_model = dnntiny(input_dim=34, class_num=2).to(device)
    DNN_model = dnnnet(input_dim=34, class_num=2).to(device)

    if training_mode == "train_test_split":
        trainDNN.trainWithsplit(DNN_model,n_frames=1, bs=32, num_epochs=1000)
        
    elif training_mode == "cross_validation":
       trainDNN.trainWithKfolds(DNN_model,kfolds=2,n_frames=1, bs=16, num_epochs=500)