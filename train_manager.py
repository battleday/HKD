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
# Helper functions
def index_probabilities(teacherProbs, batch_idx):
    """This function should batch the teacher probabilities for the L2 loss
    using the torch backend.

    Inputs
    teacherProbs: numpy array(10000 x 10); model's output probs for validation set.
    batch_idx: torch object: indices of images in batch being trained.

    Outputs
    teacherProbsBatched: torch object, teacher model output probabilities for batch
    """
    print('Not tested yet.')
    teacherProbsBatched = teacherProbs[batch_idx.to(self.device).long()]
    return teacherProbsBatched

def custom_ce(input, target, reduction):
    """Important: check this is the right ordering and where log applied.
    target: distillation target. Should be given as probabilities.
    input: model guess. Should be given as log probabilities."""
    output = - torch.sum( target * input, dim=1)
    if reduction == 'mean':
        return torch.mean(output)

class CEDivLoss(l._Loss):
    """Custom loss class for cross-entropy between soft targets and output vectors.

    Input: targets (distillation targets) as probabilities, input (model outputs) as log probabilities
    """
    __constants__ = ['reduction']

    # use mean as default reduction
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(CEDivLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return custom_ce(input, target, reduction=self.reduction)
#-----------------------------------
# Main object class
class TrainManager(object):
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
    def __init__(self, student, teacher=None, train_loader=None, 
        test_loader=None, train_config={}):
        self.config = train_config
        self.name = train_config['trial_id']
        self.device = self.config['device']
        self.student = student
        self.teacher_name = teacher['name']
        self.teacher_model = teacher['model'].cuda()
        self.have_teacher = bool(self.teacher_model)

        # set teacher to correct mode (eval)
        if self.have_teacher:
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
        trial_id = self.config['trial_id']
        lambda_ = self.config['lambda_']
        T_h = self.config['temperature_h']
        T_t = self.config['temperature_t']
        gamma_ = self.config['gamma_']

        # to record losses afteer each training epoch; will be saved
        validation_losses=[]
        training_losses=[]
        best_acc = 0
        # L1 loss criterion
        criterion = nn.CrossEntropyLoss()

        # L2 loss criterion
        if self.config['distil_fn'] == 'CE':
            print('CE model')
            distillation_criterion = CEDivLoss()
        else:
            print('{} model'.format(self.config['distil_fn']))
            distillation_criterion = nn.KLDivLoss()

        print('Starting student training, no = {} >>>>>>>>>>>>>'.format(trial_id))

        for epoch in range(epochs):
            # initialize torch model
            self.student.train()

            # loop over batches in training set (CIFAR10 validation set)
            for batch_idx, (data, target_hard, target_soft) in enumerate(self.train_loader):

                total_loss = 0
                total_loss_SL = 0
                total_loss_KD = 0
                
                # reset gradients
                self.optimizer.zero_grad()

                # organize and type data
                data = data.to(self.device).float()
                target_hard = target_hard.to(self.device).long() # may need this to survive
                target_soft = target_soft.to(self.device).float() # may need this to survive
                output = self.student(data)

                # data loss
                loss_SL = criterion(output, target_hard)
                
                # Knowledge Distillation loss
                # return teacher targets by indexing into teacherProbs with batch_idx
                if self.teacher_name == 'human':
                    # should only be the case if learning from human labels, where teacherProbs is None
                    teacher_outputs = target_soft
                elif self.teacher_name == 'baseline':
                    teacher_outputs = None
                else:
                    # IMPORTANT: CHECK THESE ARE PROBABILITIES: not here, but converted below
                    teacher_outputs = self.teacher_model(data)

                # if only training baseline teacher, use loss_SL (L1 loss) only
                if self.teacher_name == 'baseline':
                    loss_KD = loss_SL
                # i.e., if only using human labels for loss
                elif gamma_ == 1.0:
                    loss_KD = T_h * T_h * distillation_criterion(F.log_softmax(output / T_h, dim=1),
                                                 F.softmax(teacher_outputs / T_h, dim=1))
                # i.e., if only using teacher labels for loss
                elif gamma_ == 0.0:
                    loss_KD = T_t * T_t * distillation_criterion(F.log_softmax(output / T_t, dim=1),
                                                 F.softmax(teacher_outputs / T_t, dim=1))
                # if using both human and teacher
                else:
                    # convex combination of teacher and human label loss
                    loss_KD = T_t * T_t * (1-gamma_)*distillation_criterion(F.log_softmax(output / T_t, dim=1),
                                                 F.softmax(teacher_outputs / T_t, dim=1)) + T_h * T_h * gamma_ * distillation_criterion(F.log_softmax(output / T_h, dim=1),
                                                 F.softmax(target_soft / T_h, dim=1))
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
    
    def save(self, epoch, name, training_losses, validation_losses):
        trial_id = self.config['trial_id']
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
