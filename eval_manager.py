import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.modules.loss as l
from data_loader import get_cifar
from model_factory import is_resnet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np


### This script contains the main object class for training and optimization: TrainManager.
#-----------------------------------

#-----------------------------------
# Main object class
class EvalManager(object):
    """TrainManager class. Will create environment to train and save a student network.

    Inputs (input through train_student.py)
    student: torch model, to be trained
    teacher: dict, containing name (str) and probs: numpy array(10000 x 10); model's output probs for validation set.
    train_loader: ? type, CIFAR10 validation subset, already batched. From dataloader module
    test_loader: ? type, CIFAR10 training subset, already batched. From dataloader module
    train_config: dictionary, with parameters for training

    Methods
    train: train student model
    validate: take current model and calculate validation accuracy and loss
    save: dump model and results after training in dictionary form using pytorch
    adjust_learning_rate: adjusts learning rate
    """
    def __init__(self, student,  
        test_loader=None, eval_config={}):
        self.config = eval_config
        self.name = self.config['trial_id']
        self.device = self.config['device']
        self.student = student
 
        self.test_loader = test_loader
        # set up optimizer
   
    def validate(self, step=0):
        self.student.eval()
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            correct = 0
            total = 0
            acc = 0
            total_val_loss=0
            for (images,labels,_) in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.student(images)
                loss_val= criterion(outputs, labels) 
                total_val_loss+=loss_val.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
            for param_group in self.optimizer.param_groups:
                print('Learning rate:' + str(param_group['lr']))
            self.scheduler.step(acc)
            print('{{"metric": "{}_val_accuracy", "value": {}}}'.format(self.name, acc))
            return acc, total_val_loss
    
