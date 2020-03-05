import os
import pickle
import numpy as np
import torch

def load_best_model(teacher_name, master_path):
	"""Assumes teacher_name specifies a subdirectory
	of master path. Will deep scan through all subsubdirectories
	looking for best performing model."""
#	print('teacher name fed into find_best_teacher is {0}'.format(teacher_name))
	if teacher_name == 'human':
		print('human teacher')
		return {'name': 'human', 'probs': None}
	elif teacher_name == 'control':
		print('control detected but nott implemented')
	else:
		print('NO TEACHER MODEL SEARCH FUNCTION, YET')

def return_modes(dir):
    files = os.listdir(dir)
    comparisons = []
    for file in files:
        #print(file)
        file_split = file.split('_iter')
        comparisons.append(file_split[0])
    #print(comparisons)
    return comparisons

def load_torch_results(result_path):
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    temp_model = torch.load(result_path, map_location=map_location)
    working_results = temp_model['validation_losses'][-1]
    #print('model {0} at epoch {1} has val acc / loss {2}'.format(result_path, temp_model['epoch'], 
    #      working_results))
    return {'val_acc': working_results[0], 'val_loss':working_results[1]}

def average_mode(mode, dir, max_iter=9):
    accs, losses = [], []
    for iter in np.arange(max_iter+1):
        #print(iter)
        result_path = '{0}/{1}_iter_{2}_best.pth.tar'.format(dir, mode, iter)
        try:
            working_results = load_torch_results(result_path)
        except:
            continue
        accs.append(working_results['val_acc'])
        losses.append(working_results['val_loss'])
    if len(accs) == 0:
        av_acc = None
        av_loss = None
        max_ac = None
        argmax_ac = None
    else:
        av_acc = np.mean(accs)
        max_ac = np.max(accs)
        argmax_ac = np.argmax(accs)
        av_loss = np.mean(losses)
    return {'val_acc': av_acc, 'val_loss': av_loss, 'max acc': max_ac, 'argmax_ac': argmax_ac}

def print_averages(result_dict):
    for k in result_dict.keys():
        print('\n Model: {}, results: {}'.format(k, result_dict[k]))      

def compile_model_summary(dir):
    modes = return_modes(dir)
    modes.sort()
    averages = {}
    for mode in modes:
        averages[mode] = average_mode(mode, dir, max_iter=9)
        print('\n {}, {}'.format(mode, averages[mode]))
    print_averages(averages)
    return averages


