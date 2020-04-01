import os
import pickle
import numpy as np
import torch
import csv

#train.py
teacher_model = create_cnn_model(args.teacher, dataset, use_cuda=args.cuda)
		if args.teacher_checkpoint:
			print("---------- Loading Teacher -------")
			teacher_model = load_checkpoint(teacher_model, args.teacher_checkpoint)

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
parser.add_argument('--teacher-checkpoint', default='', type=str, help='optinal pretrained checkpoint for teacher')


if self.have_teacher:
			self.teacher.eval()
			self.teacher.train(mode=False)
			
		self.train_loader = train_loader
		self.test_loader = test_loader

for batch_idx, (data, target) in enumerate(self.train_loader):
				iteration += 1
				data = data.to(self.device)
				target = target.to(self.device)
				self.optimizer.zero_grad()
				output = self.student(data)
				# Standard Learning Loss ( Classification Loss)
				loss_SL = criterion(output, target) 
				loss = loss_SL
				
				if self.have_teacher:
					teacher_outputs = self.teacher(data)

teacher_name = '{}_{}_best.pth.tar'.format(args.teacher, trial_id)
teacher_model = load_checkpoint(teacher_model, os.path.join('./', teacher_name))

# Model Factory
def create_cnn_model(name, dataset="cifar100", use_cuda=False):
	"""
	Create a student for training, given student name and dataset
	:param name: name of the student. e.g., resnet110, resnet32, plane2, plane10, ...
	:param dataset: the dataset which is used to determine last layer's output size. Options are cifar10 and cifar100.
	:return: a pytorch student for neural network
	"""
	num_classes = 100 if dataset == 'cifar100' else 10
	model = None
	if is_resnet(name):
		resnet_size = name[6:]
		resnet_model = resnet_book.get(resnet_size)(num_classes=num_classes)
		model = resnet_model
		
	else:
		plane_size = name[5:]
		model_spec = plane_cifar10_book.get(plane_size) if num_classes == 10 else plane_cifar100_book.get(plane_size)
		plane_model = ConvNetMaker(model_spec)
		model = plane_model

	# copy to cuda if activated
	if use_cuda:
		model = model.cuda()
		
	return model


def is_resnet(name):
	"""
	Simply checks if name represents a resnet, by convention, all resnet names start with 'resnet'
	:param name:
	:return:
	"""
	name = name.lower()
	return name.startswith('resnet')

from resnet_cifar import *
from plain_cnn_cifar import *


def load_torch_results(model_path):
    """Given the path to a torch model, load and return it.

    Inputs
    model_path: str, path to torch model

    Outputs
    model: torch model, loaded from model_path
    """
    # check hardware
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    model = torch.load(model_path, map_location=map_location)
    return model

def load_best_model(teacher_name, master_path, optional_args=None):
    """If given a path, this function will return the output probabilities of 
    the best teacher under that path as a numpy array.

    Inputs
    teacher_name: str, gives teacher name
    master_path: str, gives path where teacher's name will be name of a directory below
                 with all relevant model results in it.

    Outputs
    teacher: dict, containing name (str), and probs, a 10000 x 10 numpy array with teacher network's output probabilities
                in order for validation subset of CIFAR10
    """
    
    #If the teacher is human (i.e., if pretraining), return None (dataloader handles 
    #human soft labels).
    if teacher_name == 'human':
        print('human teacher')
        teacher = {'name': 'human', 'probs': None}
    elif teacher_name == 'baseline':
        print('human teacher')
        teacher = {'name': 'baseline', 'probs': None}
    else:
        print('teacher is: {0}'.format(teacher_name))

        #make sure best model is up to date at time of load
        new_path = save_best_model('{0}/{1}'.format(master_path, teacher_name, optional_args))

        #generate probabilities under that model
        teacher = load_torch_probabilities(new_path)
    return teacher

def save_best_model(results_path, optional_args):
    """Will deep scan through all subsubdirectories
    looking for best performing model given a path and some options.

    Will copy and rename these models, and return the new path.

    Inputs
    results_path: str, master directory to scan under

    Outputs
    model_path
    Needs to be implemented
    """
    print("not implemented yet")
    return '{0}/best_teacher.pth.tar'.format(results_path)


def load_torch_probabilities(model_path):
    """Given a path, will load torch model return its output probabilities for 
    the validation subset of CIFAR10

    Inputs
    model_path: str, path to model

    Outputs
    teacherProbs: numpy array (10000 x 10); model's output probs for validation set.
    """

    # load the model
    model = load_torch_results(model_path)

    # generate model predictions
    teacherProbs = validation_probabilities(model)

    return teacherProbs


def validation_probabilities(model):
    """Will take saved model, and compute the model probabilities for the 
    validation subset of CIFAR10

    Input
    model: torch model, which we want to compute probabilities from

    Output
    teacherProbs: numpy array(10000 x 10); model's output probs for validation set.
    """
    print("Not implemented yet")

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

