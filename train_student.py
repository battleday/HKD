import os
# import nni
import copy
import pickle
import torch
import argparse
from data_loader import get_cifar
from find_best_teacher import load_best_model
from train_manager import TrainManager
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
	parser = argparse.ArgumentParser(description='Training KD Teachers Code')
	parser.add_argument('--master_outdir', default='', type=str, help='model dump dir')
	parser.add_argument('--master_architecture', default='resnet', type=str, help='next level down from model dump')
	parser.add_argument('--teacher', default='human', type=str, help='teacher name')
	parser.add_argument('--student', default='resnet8', type=str, help='student name')
	parser.add_argument('--manual_seed', default=0, type=int, help='manual seed')
	parser.add_argument('--iter', default=0, type=int, help='run # with same parameters')

	parser.add_argument('--temperature_t', default=1, type=int,  help='student teacher temperature')
	parser.add_argument('--temperature_h', default=1, type=int,  help='student human temperature')
	parser.add_argument('--lambda_', default=0.5, type=float,  help='weighted average')
	parser.add_argument('--gamma_', default=0.0, type=float,  help='weighted average')
	parser.add_argument('--distil_fn', default='KD', type=str,  help='for distillation loss (KD or CE)')

	parser.add_argument('--epochs', default=500, type=int,  help='number of total epochs to run')
	parser.add_argument('--dataset', default='cifar10', type=str, help='dataset. can be either cifar10 or cifar100')
	parser.add_argument('--batch-size', default=128, type=int, help='batch_size')
	parser.add_argument('--learning-rate', default=0.1, type=float, help='initial learning rate')
	parser.add_argument('--momentum', default=0.9, type=float,  help='SGD momentum')
	parser.add_argument('--weight-decay', default=1e-4, type=float, help='SGD weight decay (default: 1e-4)')
	
	parser.add_argument('--cuda', default=False, type=str2bool, help='whether or not use cuda(train on GPU)')
	parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')
	parser.add_argument('--trial_id', default='', type=str,  help='id string')
	
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	# Parse arguments 
	args = parse_arguments()
	print(args)

	# prepare paths and log
	# student results will be saved under their teacher's superdirectory
	student_dir = '{}/teacher_{}'
	save_path = '{}/{}/{}'.format(args.master_outdir, args.master_architecture, args.teacher)
	log_path = '{}/training_log.log'.format(save_path)
	print('saving model run in {} \n'.format(save_path))

	if not os.path.exists(save_path):
    		print('making new dirs')
    		os.makedirs(save_path)

	logs = open(log_path, 'a')

	# set seed
	torch.manual_seed(args.manual_seed)
	torch.cuda.manual_seed(args.manual_seed)
	
	# prepare dataset-related hyperparameters
	dataset = args.dataset
	if dataset == 'cifar10':
		num_classes = 10 
	else:
		print('cifar10 not loaded!')

	# dataloaders for training paradigm
	train_loader, test_loader = get_cifar(num_classes, batch_size=args.batch_size, crop=True)

	# prepare train_config, to be passed into TrainManager class
	train_config = {
				'epochs': args.epochs,
				'learning_rate': args.learning_rate,
				'momentum': args.momentum,
				'weight_decay': args.weight_decay,
				'device': 'cuda' if args.cuda else 'cpu',
				'trial_id': args.trial_id,
				'batch_size': args.batch_size,
				'distil_fn': args.distil_fn,
				'lambda_': args.lambda_,
				'temperature_h': args.temperature_h,
				'temperature_t': args.temperature_t,
				'gamma_': args.gamma_
			}

	# create student model for CIFAR10; usually shake26 for initial student and resnet8 thereafter.
	student_model = create_cnn_model(args.student, dataset, use_cuda=args.cuda)

	# below will be dict with teacher name and probs for CIFAR10 validation subset
	teacher_model = load_best_model(args.teacher, args.master_outdir, optional_arguments = None)

    # unique identifier
	student_name = 'student_{0}_distil_fn_{1}_temperature_h_{2}_temperature_t_{3}_lambda_{4}_gamma_{5}_iter_{6}_best.pth.tar'.format(args.student, 
								args.distil_fn, args.temperature_h, args.temperature_t, args.lambda_, args.gamma_, args.iter)

    # where to dump final model
	train_config['outfile'] = '{}/{}'.format(save_path, student_name)

	print("---------- Training Student -------")
	
	student_trainer = TrainManager(student_model, teacherProbs=teacher_model['probs'], 
		                           train_loader=train_loader, test_loader=test_loader, train_config=train_config)

	best_valacc, best_valloss = student_trainer.train()

	print("Best student accuacy for teacher {0}, student {1}, distil_fn {2}, temp {3}, lambda {4}, gamma {5}, iter {6} is {7}".format(args.teacher,
		args.student, args.iter, args.temperature, args.lambda_, args.gamma_, args.iter, best_valacc))

	logline = "teacher {0}, student {1}, distil_fn {2}, temp {3}, lambda {4}, gamma {5}, iter {6},valacc {7}, valloss {8}, \n".format(args.teacher,
		      args.student, args.distil_fn, args.temperature, args.lambda_, args.gamma_, args.iter, best_valacc, best_valloss)
				
	logs.write(logline)
	logs.flush()
	
	os.fsync(logs.fileno())
