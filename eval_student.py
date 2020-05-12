import os
# import nni
import copy
import pickle
import torch
import argparse
from data_loader import get_cifar
from imagenet_far_dataloader import get_imagenet
from cinic_dataloader import get_cinic
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
    parser.add_argument('--student_name', default='resnet8', type=str, help='student name')
    parser.add_argument('--manual_seed', default=0, type=int, help='manual seed')

    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset. can be either cifar10 or cifar100')
    parser.add_argument('--batch-size', default=128, type=int, help='batch_size')

    parser.add_argument('--cuda', default=True, type=str2bool, help='whether or not use cuda(train on GPU)')
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

    # dataloaders for training paradigm --- do all three for generalization
    _, test_loader_cifar = get_cifar(num_classes, batch_size=args.batch_size, 
                                     crop=True)
    _, test_loader_imagenet_far = get_imagenet(num_classes, 
                                  batch_size=args.batch_size, crop=True) 

    _, test_loader_cinic = get_cinic(num_classes, 
                                  batch_size=args.batch_size, crop=True) 
    # prepare train_config, to be passed into TrainManager class
    eval_config = {
                'device': 'cuda' if args.cuda else 'cpu',
                'batch_size': args.batch_size,
                'validate': 'index',
                'cinic': False
            }
    student_path = '{}/{}'.format(args.master_outdir, args.student_name)
    if os.path.exists('{}_generalization.npy'.format(student_path[:-8])):
       print('existing model found')
       exit()    
    print('student path is: {}'.format(student_path))
    # create student model for CIFAR10; usually shake26 for initial student and resnet8 thereafter.
    student_model = load_teacher_model(student_path, 'resnet8', dataset, cuda_option=args.cuda)

    # where to dump final model
    
    print("---------- Evaluating Student -------")
    student_eval_cifar = EvalManager(student_model, test_loader=test_loader_cifar,
                                     eval_config=eval_config)

    best_valacc1, best_valloss1 = student_eval_cifar.validate()

    print('valacc cifar: {}, vallos cifar: {}'.format(best_valacc1, best_valloss1))

    eval_config['cinic'] = True

    student_eval_cinic = EvalManager(student_model, test_loader=test_loader_cinic,
                                     eval_config=eval_config)

    best_valacc2, best_valloss2 = student_eval_cinic.validate()

    print('valacc cinic: {}, vallos cinic: {}'.format(best_valacc2, best_valloss2))

    student_eval_imagenet = EvalManager(student_model, 
                test_loader=test_loader_imagenet_far, eval_config=eval_config)

    best_valacc3, best_valloss3 = student_eval_imagenet.validate()

    print('valacc imagenet: {}, vallos imagenet: {}'.format(best_valacc3, best_valloss3))
    results = np.array([[best_valacc1, best_valacc2, best_valacc3],[best_valloss1, best_valloss2, best_valloss3]])
    np.save('{}_generalization.npy'.format(student_path[:-8]), results)    

