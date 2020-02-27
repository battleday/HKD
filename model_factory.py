from architectures.resnet_cifar import *
from architectures.plain_cnn_cifar import *
from architectures.resnext import *
from architectures.shake_shake import *
from architectures.resnet import *
from architectures.wrn import *

def is_resnet(name):
	"""
	Simply checks if name represents a resnet, by convention, all resnet names start with 'resnet'
	:param name:
	:return:
	"""
	name = name.lower()
	return name.startswith('resnet')

def is_resnext(name):
	"""
	Simply checks if name represents a resnet, by convention, all resnet names start with 'resnet'
	:param name:
	:return:
	"""
	name = name.lower()
	return name.startswith('resnext')

def is_shake(name):
	"""
	Simply checks if name represents a resnet, by convention, all resnet names start with 'resnet'
	:param name:
	:return:
	"""
	name = name.lower()
	return name.startswith('shake')

def is_wrn(name):
	"""
	Simply checks if name represents a resnet, by convention, all resnet names start with 'resnet'
	:param name:
	:return:
	"""
	name = name.lower()
	return name.startswith('wrn')


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
		resnet_model = Network2(resnet_size)
		model = resnet_model
	
	elif is_resnext(name):
		resnext_size = name[7:]
		resnext_model = Network()
		model = resnext_model
		
	elif is_shake(name):
		shake_size = name[5:]
		shake_model = Network1()
		model = shake_model
		
	elif is_wrn(name):
		
		wrn_model = Network3()
		model = wrn_model

	else:
		plane_size = name[5:]
		model_spec = plane_cifar10_book.get(plane_size) if num_classes == 10 else plane_cifar100_book.get(plane_size)
		plane_model = ConvNetMaker(model_spec)
		model = plane_model

	# copy to cuda if activated
	if use_cuda:
		model = model.cuda()
		
	return model
