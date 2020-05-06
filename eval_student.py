import os
# import nni
import copy
import pickle
import torch
import argparse
from data_loader import get_cifar
from find_best_teacher import *
from eval_manager import EvalManager
from model_factory import create_cnn_model, is_resnet

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    else:
        return False
    
def load_checkpoint(model, checkpoint_path):
    """
    Loads weights from checkpoint
    :param model: a pytorch nn student
    :param str checkpoint_path: address/path of a file
    :return: pytorch nn student with weights loaded from checkpoint
    """
    model_ckp = torch.load(checkpoint_path)
    model.load_state_dict(model_ckp['model_state_dict'])
    return model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluating student generalization Code')
    parser.add_argument('--master_outdir', default='', type=str, help='model dump dir')
    parser.add_argument('--student', default='resnet8', type=str, help='student name')
    parser.add_argument('--manual_seed', default=0, type=int, help='manual seed')

    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset. can be either cifar10 or cifar100')
    parser.add_argument('--batch-size', default=128, type=int, help='batch_size')

    parser.add_argument('--cuda', default=False, type=str2bool, help='whether or not use cuda(train on GPU)')
    parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse arguments 
    args = parse_arguments()
    print(args)

    # prepare paths and log
    # student results will be saved under their teacher's superdirectory

    # set seed
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    
    # prepare dataset-related hyperparameters
    dataset = args.dataset
    if dataset == 'cifar10':
        num_classes = 10 
    else:
        print('cifar10 not loaded!')
        return

    # dataloaders for training paradigm --- do all three for generalization
    _, test_loader = get_cifar(num_classes, batch_size=args.batch_size, crop=True)

    # prepare train_config, to be passed into TrainManager class
    eval_config = {
                'device': 'cuda' if args.cuda else 'cpu',
                'trial_id': args.trial_id,
                'batch_size': args.batch_size
            }

    # create student model for CIFAR10; usually shake26 for initial student and resnet8 thereafter.
    student_model = load_teacher_model(arg.student_path, 'resnet8', dataset, use_cuda=args.cuda)

    # where to dump final model
    
    print("---------- Evaluating Student -------")
    student_trainer = EvalManager(student_model, test_loader=test_loader, eval_config=eval_config)

    best_valacc, best_valloss = student_trainer.validate()


