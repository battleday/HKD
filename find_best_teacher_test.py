import os
import pickle
import numpy as np
import torch
import csv
from model_factory import *


def load_checkpoint(model, checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn model
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn student with weights loaded from checkpoint
	"""
	model_ckp = torch.load(checkpoint_path)
	model.load_state_dict(model_ckp['model_state_dict'])
	return model

def load_teacher_model(path, teacher_arch, dataset='cifar10', cuda_option = False):
    teacher_model = create_cnn_model(teacher_arch, dataset, use_cuda=cuda_option)
    teacher_model = load_checkpoint(teacher_model, path)
    return teacher_model

def load_best_model(teacher, master_path, cuda_option = False):
    """If given a path, this function will return the output probabilities of 
    the best teacher under that path as a numpy array.

    Inputs
    teacher: dict {'name': , 'arch'}, gives teacher name and master architecture
    master_path: str, gives path where teacher's name will be name of a directory below
                 with all relevant model results in it.

    Outputs
    teacher: dict, containing name (str), and probs, a 10000 x 10 numpy array with teacher network's output probabilities
                in order for validation subset of CIFAR10
    """
    
    #If the teacher is human (i.e., if pretraining), return None (dataloader handles 
    #human soft labels).
    if teacher['name'] == 'human':
        print('human teacher')
        teacher_model = None
    elif teacher['name'] == 'baseline':
        print('no teacher')
        teacher_model = None
    else:
        print('teacher is: {0}'.format(teacher['name']))

        #make sure best model is up to date at time of load
        teacher_path = '{0}/{1}_{2}/best_teacher.pth.tar'.format(master_path, 
                                                             teacher['name'],
                                                             teacher['args'])

        #generate probabilities under that model
        teacher = load_teacher_model(teacher_path, teacher['arch'], cuda_option)
    return teacher


def return_modes(results_path):
    """Given a directory, will return a sorted list of all combinations of 
    all types of model saved under it. This abstracts over iterations / runs
    with the same learning parameters.

    Inputs
    results_path: str, master directory to search under

    Outputs
    comparisons: list, sorted list of trained model types
    """

    # find all files under path 
    files_raw = os.listdir(results_path)

    # only record those with the right extension
    files = [x for x in files_raw if '.pth' in x]
    comparisons = []
    for file in files:
        # ignore run number / iter
        file_split = file.split('_iter')
        comparisons.append(file_split[0])
    return sorted(comparisons)


def return_validation_stats(torch_model):
    """Given a torch model, return stats of interest.
    Inputs
    torch_model: torch model, loaded

    Outputs
    result_dict: dictionary, containing last-recorded validation accuracy and loss
                 prior to model dump. If this is best epoch, will be best 
    #stats under model. 
    """
    #last validation losses computed. 
    temp = torch_model['validation_losses'][-1]
    result_dict = {'val_acc': temp[0], 'val_loss': temp[1]}
    return result_dict

def average_mode(mode, result_dir, max_iter=9):
    """this function averages over iters for a given combination of training parameters.
    
    Inputs
    mode: str, combination of learning parameters that specify one model type
    dir: str, location in which to find model files
    max_iter: int, the number of runs

    Outputs
    result_dict: dictionary, containing the averaged results for one model type,
                 including score and index of best model.
    """
    accs, losses = [], []
    # scan over all iters; must add one
    for iter in np.arange(max_iter + 1):
        result_path = '{0}/{1}_iter_{2}_best.pth.tar'.format(result_dir, mode, iter)
        try:
            working_results = load_torch_results(result_path)
            working_results = return_validation_stats(working_results)
        except Exception as e:
            print('Result path invalid or not torch model. Dir: {0}; Mode: {1}; error: {2}'.format(mode, e, result_dir))
            continue
        accs.append(working_results['val_acc'])
        losses.append(working_results['val_loss'])

    av_acc = np.mean(accs)
    max_ac = np.max(accs)
    av_loss = np.mean(losses)
    # best individual model 
    argmax_ac = np.argmax(accs)

    result_dict = {'val_acc': av_acc, 'val_loss': av_loss, 'max acc': max_ac, 'best_iter': argmax_ac}
    return result_dict

def print_averages(result_dict):
    """This is a function to print a result dict. Should be extended to 
    dump a csv.
    """
    for k in result_dict.keys():
        print('\n Model: {}, results: {}'.format(k, result_dict[k]))      

def compile_model_summary(results_path):
    """For a given result directory, compile and print a summary 
    of the results of all model types within it.
    
    Inputs
    results_path: str, master directory to search under

    Outputs
    averages: dictionary, containing averages by modes
    """

    # gives a sorted list of different model types
    modes = return_modes(results_path)
    #print("modes are: {0}".format([x + '\n' for x in modes]))
    averages = {}
    for mode in modes:
        averages[mode] = average_mode(mode, results_path, max_iter=9)
#        print('\n {}, {}'.format(mode, averages[mode]))
    # below function currently prints, but could instead dump to csv for better readability
    print_averages(averages)
    return averages


def integrate_hyperparameters(model_name, result_dict):
    """Takes a model name and splits into hyperparameters, 
    to be saved in result dict and be returned"""
    hyperparameters = ['distil_fn', 'temperature_h', 
                       'temperature_t', 'lambda', 'gamma']
    for h in hyperparameters:
        split_string = model_name.split(h)
        split_split_string = split_string[1].split('_')
        result = split_split_string[1]
        #print('{}: {}'.format(h, result))
        result_dict[h] = result
    return result_dict

def csv_dump(average_dict, out_dir):
    """Takes a dictionary of model averages, and dumps it to a csv"""
    modelList = list(average_dict.keys())
    print('modelList[:5] is {}'.format(modelList[:5]))
    modelListSorted = sorted(modelList, key=lambda x: float(x.split('lambda_')[1].split('_')[0]))
    print('modelListSorted[:5] is {}'.format(modelListSorted[:5]))
    #print(modelList)
    newAverageList = [integrate_hyperparameters(x, 
                        average_dict[x]) for x in modelListSorted]
    csv_columns = ['distil_fn', 'lambda', 'temperature_h', 'temperature_t', 'gamma',
                   'val_acc', 'val_loss', 'max acc', 'best_iter', 'val_acc']
    csv_file = '{}/average_results.csv'.format(out_dir)
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in newAverageList:
                #print(data)
                writer.writerow(data)
    except IOError:
        print("I/O error")

