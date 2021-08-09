import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=0.25, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight) 
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none') # important to add reduction='none' to keep per-batch-item loss
pt = torch.exp(-ce_loss)
focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean() # mean over the batch