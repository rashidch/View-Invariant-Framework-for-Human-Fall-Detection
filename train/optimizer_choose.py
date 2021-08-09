from __future__ import print_function, division

import torch
import shutil
import inspect


def optimizer_choose(model,optimizer):
    params = []
    for key, value in model.named_parameters():
        if value.requires_grad:
            params += [{'params': [value], 'lr':0.01, 'key': key, 'weight_decay':0.005}]
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(params)
        
    elif optimizer == 'sgd':
        momentum = 0.9
        optimizer = torch.optim.SGD(params, momentum=momentum)
       
    elif optimizer == 'sgd_nev':
        momentum = 0.9
        optimizer = torch.optim.SGD(params, momentum=momentum, nesterov=True)
        
    else:
        momentum = 0.9
        optimizer = torch.optim.SGD(params, momentum=momentum)
    
    return optimizer

def optimizer_choose2(model1, model2,optimizer):
    params1 = []
    params2 = []
    for key, value in model1.named_parameters():
        if value.requires_grad:
            params1 += [{'params': [value], 'lr':0.001, 'key': key, 'weight_decay':5e-3}]
    
    for key, value in model2.named_parameters():
        if value.requires_grad:
            params2 += [{'params': [value], 'lr':0.001, 'key': key, 'weight_decay':0}]

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(params1+params2)
        
    elif optimizer == 'sgd':
        momentum = 0.9
        optimizer = torch.optim.SGD(params1+params2, momentum=momentum)
       
    elif optimizer == 'sgd_nev':
        momentum = 0.9
        optimizer = torch.optim.SGD(params1+params2, momentum=momentum, nesterov=True)
        
    else:
        momentum = 0.9
        optimizer = torch.optim.SGD(params, momentum=momentum)
    
    return optimizer