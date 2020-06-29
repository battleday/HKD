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
torch.set_printoptions(precision=10)

### This script contains the main object class for training and optimization: TrainManager.
#-----------------------------------
# Helper functions

def custom_ce(input, target, reduction):
    """This is a custom cross-entropy function, as pytorch indexes into a one-hot for 
    its cross-entropy loss.Expects target to be probabilities,
    input to already be log probabilities."""
    output = - torch.sum( target * input, dim=1)
    if reduction == 'mean':
        return torch.mean(output)

class CEDivLoss(l._Loss):
    """Custom loss class for cross-entropy between soft targets and output vectors.
    Input: targets (distillation targets) as probabilities, input (model outputs) as log probabilities
    to mirror functional form above in custom_ce"""
    __constants__ = ['reduction']

    # use mean as default reduction
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(CEDivLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return custom_ce(input, target, reduction=self.reduction)
#-----------------------------------
# Main object class
class TrainManager(object):
    """TrainManager class. Defines environment to train and save a student network.

    Called by train_student.py

    Inputs 
    student: torch model, to be trained
    teacher: dict, containing name (str) and model (pytorch)
    train_loader: ? type, CIFAR10 validation subset, already batched. From dataloader module
    test_loader: ? type, CIFAR10 training subset, already batched. From dataloader module
    train_config: dictionary, with parameters for training

    Methods
    train: train student model
    validate: take current model and calculate validation accuracy and loss
    save: dump model and results after training in dictionary form using pytorch
    adjust_learning_rate: adjusts learning rate
    """
    def __init__(self, student, teacher=None, train_loader=None, 
        test_loader=None, train_config={}):
        self.config = train_config
        self.device = self.config['device']
        self.student = student
        self.teacher_name = teacher['name']
        if self.teacher_name == 'baseline' or self.teacher_name =='human':
            self.have_teacher = False
        else:
            self.have_teacher = True

        # set teacher to correct mode (eval)
        if self.have_teacher:
            self.teacher_model = teacher['model'].cuda()
            self.teacher_model.eval()
            self.teacher_model.train(mode=False)
        
        self.train_loader = train_loader
        self.test_loader = test_loader

        # set up optimizer
        self.optimizer = optim.SGD(self.student.parameters(),
                                   lr=self.config['learning_rate'],
                                   momentum=self.config['momentum'],
                                   weight_decay=self.config['weight_decay'])

        # set up scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', factor = 0.9, patience = 5, 
            verbose = True)
            
    def train(self):
        epochs = self.config['epochs']
        lambda_ = self.config['lambda_']
        T_h = self.config['temperature_h']
        T_t = self.config['temperature_t']
        gamma_ = self.config['gamma_']

        # to record losses afteer each training epoch; will be saved
        validation_losses=[]
        training_losses=[]
        best_acc = 0

        # Standard loss criterion ("L1"; i.e., for error with one-hots)
        criterion = nn.CrossEntropyLoss()

        # Distillation loss criterion ("L2")
        if self.config['distil_fn'] == 'CE': # if cross-entropy
            print('CE model')
            distillation_criterion = CEDivLoss()
        else: # if KL divergence
            print('{} model'.format(self.config['distil_fn']))
            distillation_criterion = nn.KLDivLoss()

        print('Starting student training, put your feet up ;) >>>>>>>>>>>>>')

        for epoch in range(epochs):

            # initialize student model
            self.student.train()

            # loop over batches in training set (CIFAR10 validation set)
            for batch_idx, (data, target_hard, target_soft) in enumerate(self.train_loader):
                # data are image pixels, target_hard is one-hot, target_soft is human probabilities.
                # N.B. it is imperative that entries of teacher_soft are probabilities

                total_loss = 0
                total_loss_SL = 0 # standard loss
                total_loss_KD = 0 # knowledge distillation loss
                
                # reset gradients
                self.optimizer.zero_grad()

                # organize and type data
                data = data.to(self.device).float()
                target_hard = target_hard.to(self.device).long() # as torch cross-entropy loss indexes
                target_soft = target_soft.to(self.device).float() # as custom ce uses probability vectors

                # now that we are taking log of human guesses, must use smoothing. 
                # One guess has about 0.02 probability mass, so we can use laplace smoothing as follows:
                target_soft = (target_soft + 0.02) / 1.2  # as there are 10 categories)
                output = self.student(data)

                # data loss
                loss_SL = criterion(output, target_hard)
                
                # Knowledge Distillation loss; NB, have tried to correct
                if self.teacher_name == 'human':
                    # should only be the case if learning from human labels
                    teacher_outputs = torch.log(target_soft)
                    
                    # first argument is target and should be probability. Second argument is model output
                    # and should be log probability
                    loss_KD = (T_h ** 2) * distillation_criterion(F.log_softmax(output / T_h, dim=1), 
                                                              F.softmax(teacher_outputs / T_h, dim=1))
                elif self.teacher_name == 'baseline':
                    # only the case for training the first teacher
                    teacher_outputs = None
                    loss_KD = loss_SL
                else:
                    # this is for training students, where teacher name will be a conjunction.
                    teacher_outputs = self.teacher_model(data) # should be logits
                    human_outputs = torch.log((target_soft + 0.02) / 1.2) # smooth and convert to log probabilities

                    # ordering as above. Gamma = 1 will give only human label. Gamma = 0 will give only teacher (classic KD)
                    teacher_term = (1-gamma_) * (T_t **2) * distillation_criterion(F.log_softmax(output / T_t, dim=1), 
                                                                               F.softmax(teacher_output / T_t, dim=1))
                    human_term = gamma * (T_h **2) * distillation_criterion(F.log_softmax(output / T_h, dim=1), 
                                                                         F.softmax(human_outputs / T_h, dim=1))

                    loss_KD = teacher_term + human_term

                # total loss
                loss = (1 - lambda_) * loss_SL + lambda_ * loss_KD
                total_loss += loss
                total_loss_SL += loss_SL
                total_loss_KD += loss_KD
                loss.backward()
                self.optimizer.step()

            # group training losses
            training_losses.append((total_loss, total_loss_SL, total_loss_KD))
            
            # calculate validation losses
            val_acc, val_loss = self.validate(step=epoch)
            validation_losses.append((val_acc, val_loss))

            print("epoch {}/{} done \n".format(epoch, epochs))          
            print("training loss epoch>>",total_loss)
            print("validation accuracy and loss of epoch>>",val_acc, val_loss)
            
            # only save if best epoch so far: if so, overwrite model
            if val_acc > best_acc:
                best_acc = val_acc
                best_loss = val_loss
                self.save(epoch, self.config['outfile'], training_losses, validation_losses)

        return best_acc, best_loss
    
    def validate(self, step=0):
        self.student.eval()
        # as we are only doing validation loss on hard labels, can use inbuilt loss function
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
            print('{{"metric": "val_accuracy", "value": {}}}'.format(acc))
            return acc, total_val_loss
    
    def save(self, epoch, name, training_losses, validation_losses):
        """will save model and parameters in path <name>"""
        torch.save({
                'model_state_dict': self.student.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'training_losses': training_losses,
                'validation_losses': validation_losses
            }, name)

    def adjust_learning_rate(self, optimizer, epoch):
        epochs = self.config['epochs']
        models_are_plane = self.config['is_plane']
        
        # depending on dataset
        if models_are_plane:
            lr = 0.01
        else:
            lr= self.config['learning_rate']
            if epoch < int(epochs/2.0):
                lr = 0.096
            elif epoch < int(epochs*3/4.0):
                lr = 0.096 * 0.1
            else:
                lr = 0.096 * 0.01
        
        print('Learning rate >>>',lr)
        # update optimizer's learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
