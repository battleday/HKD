


++ b/shake_shake.py
@ -0,0 +1,188 @@
import torch
import torch.nn as nn
import torch.nn.functional as F

from shake_function import get_alpha_beta, shake_function


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class ResidualPath(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualPath, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(x, inplace=False)
        x = F.relu(self.bn1(self.conv1(x)), inplace=False)
        x = self.bn2(self.conv2(x))
        return x


class DownsamplingShortcut(nn.Module):
    def __init__(self, in_channels):
        super(DownsamplingShortcut, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.conv2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn = nn.BatchNorm2d(in_channels * 2)

    def forward(self, x):
        x = F.relu(x, inplace=False)
        y1 = F.avg_pool2d(x, kernel_size=1, stride=2, padding=0)
        y1 = self.conv1(y1)

        y2 = F.pad(x[:, :, 1:, 1:], (0, 1, 0, 1))
        y2 = F.avg_pool2d(y2, kernel_size=1, stride=2, padding=0)
        y2 = self.conv2(y2)

        z = torch.cat([y1, y2], dim=1)
        z = self.bn(z)

        return z


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, shake_config):
        super(BasicBlock, self).__init__()

        self.shake_config = shake_config

        self.residual_path1 = ResidualPath(in_channels, out_channels, stride)
        self.residual_path2 = ResidualPath(in_channels, out_channels, stride)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('downsample',
                                     DownsamplingShortcut(in_channels))

    def forward(self, x):
        x1 = self.residual_path1(x)
        x2 = self.residual_path2(x)

        if self.training:
            shake_config = self.shake_config
        else:
            shake_config = (False, False, False)

        alpha, beta = get_alpha_beta(x.size(0), shake_config, x.device)
        y = shake_function(x1, x2, alpha, beta)

        return self.shortcut(x) + y


class Network1(nn.Module):
    def __init__(self):
        super(Network1, self).__init__()

        input_shape = (1,3,32,32)
        n_classes = 10

        base_channels = 64
        depth = 26
        self.shake_config = (True, True,
                             True)

        block = BasicBlock
        n_blocks_per_stage = (depth - 2) // 6
        assert n_blocks_per_stage * 6 + 2 == depth

        n_channels = [base_channels, base_channels * 2, base_channels * 4]

        self.conv = nn.Conv2d(
            input_shape[1],
            n_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn = nn.BatchNorm2d(base_channels)

        self.stage1 = self._make_stage(
            n_channels[0], n_channels[0], n_blocks_per_stage, block, stride=1)
        self.stage2 = self._make_stage(
            n_channels[0], n_channels[1], n_blocks_per_stage, block, stride=2)
        self.stage3 = self._make_stage(
            n_channels[1], n_channels[2], n_blocks_per_stage, block, stride=2)

        # compute conv feature size
        with torch.no_grad():
            self.feature_size = self._forward_conv(
                torch.zeros(*input_shape)).view(-1).shape[0]

        self.fc = nn.Linear(self.feature_size, n_classes)

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                stage.add_module(
                    block_name,
                    block(
                        in_channels,
                        out_channels,
                        stride=stride,
                        shake_config=self.shake_config))
            else:
                stage.add_module(
                    block_name,
                    block(
                        out_channels,
                        out_channels,
                        stride=1,
                        shake_config=self.shake_config))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
 No newline at end of file
iff --git a/test50k_images.npy b/test50k_images.npy
ew file mode 100644
ndex 0000000..2ccd076
inary files /dev/null and b/test50k_images.npy differ
iff --git a/test50k_labels.npy b/test50k_labels.npy
ew file mode 100644
ndex 0000000..9b781d2
inary files /dev/null and b/test50k_labels.npy differ
iff --git a/train.py b/train.py
ew file mode 100644
ndex 0000000..d485514
-- /dev/null
++ b/train.py
@ -0,0 +1,341 @@
import os
# import nni
import copy
import pickle
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_loader import get_cifar
from model_factory import create_cnn_model, is_resnet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# import pdb; pdb.set_trace()

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	else:
		return False
	
	
def parse_arguments():
	parser = argparse.ArgumentParser(description='TA Knowledge Distillation Code')
	parser.add_argument('--epochs', default=500, type=int,  help='number of total epochs to run')
	parser.add_argument('--dataset', default='cifar100', type=str, help='dataset. can be either cifar10 or cifar100')
	parser.add_argument('--batch-size', default=128, type=int, help='batch_size')
	parser.add_argument('--learning-rate', default=0.1, type=float, help='initial learning rate')
	parser.add_argument('--momentum', default=0.9, type=float,  help='SGD momentum')
	parser.add_argument('--weight-decay', default=1e-4, type=float, help='SGD weight decay (default: 1e-4)')
	parser.add_argument('--teacher', default='', type=str, help='teacher student name')
	parser.add_argument('--student', '--model', default='resnet8', type=str, help='teacher student name')
	parser.add_argument('--teacher-checkpoint', default='', type=str, help='optinal pretrained checkpoint for teacher')
	parser.add_argument('--cuda', default=False, type=str2bool, help='whether or not use cuda(train on GPU)')
	parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')
	parser.add_argument('--temperature', default=1, type=int,  help='student temperature')
	parser.add_argument('--lambda_', default=0.5, type=float,  help='weighted average')
	parser.add_argument('--gamma_', default=1, type=float,  help='weighted combination')
	parser.add_argument('--trial_id', default=1, type=int,  help='id number')
	
	args = parser.parse_args()
	return args


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


class TrainManager(object):
	def __init__(self, student, teacher=None, train_loader=None, test_loader=None, train_config={}):
		self.student = student
		self.teacher = teacher
		self.have_teacher = bool(self.teacher)
		self.device = train_config['device']
		self.name = train_config['name']
		
		
		self.optimizer = optim.SGD(self.student.parameters(),
								   lr=train_config['learning_rate'],
								   momentum=train_config['momentum'],
								   weight_decay=train_config['weight_decay'])
		# self.optimizer = optim.Adam(self.student.parameters(),
		# 						   lr=train_config['learning_rate'],
		# 						   )
		self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', factor = 0.9, patience = 5, verbose = True)
		if self.have_teacher:
			self.teacher.eval()
			self.teacher.train(mode=False)
			
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.config = train_config
	
	def train(self):
		# lambda_ = self.config['lambda_student']
		# T = self.config['T_student']

		lambda_ = self.config['lambda_']
		gamma_ = self.config['gamma_']
		T = self.config['temperature']

		epochs = self.config['epochs']
		trial_id = self.config['trial_id']
		
		max_val_acc = 0
		iteration = 0
		best_acc = 0
		criterion = nn.CrossEntropyLoss()
		criterion1 = nn.MSELoss()
		validation_losses=[]
		print('Starting Student training, no = %s with gamma= %s, lambda= %s, temperature= %s >>>>>>>>>>>>>'%(trial_id,gamma_,lambda_,T))
		for epoch in range(epochs):
			self.student.train()
			# self.adjust_learning_rate(self.optimizer, epoch)
			total_loss=0
			loss = 0
			for batch_idx, (data, target_hard,target_soft) in enumerate(self.train_loader):
				
				iteration += 1
				
				data = data.to(self.device).float()
				target_hard = target_hard.to(self.device).long()
				target_soft = target_soft.to(self.device).float()
				# print(target_hard)

				self.optimizer.zero_grad()
				output = self.student(data)
				# Standard Learning Loss ( Classification Loss)
				# loss_SL = criterion(output, target_hard)
				ce_fits = torch.mean(torch.sum(- target_soft * F.log_softmax(output, dim=1), dim=1))
				total_loss+=ce_fits 
				loss_SL = torch.mean(torch.sum(- target_soft * F.log_softmax(output, dim=1), dim=1))
				# loss_SL = 0.8*loss_SL1 + 0.2*loss_SL2
				diverged_examples= []
				non_diverged_examples = []
				if self.have_teacher:
					teacher_outputs1= self.teacher(data)
					# print(teacher_outputs[0],teacher_outputs[1],teacher_outputs[2])
					teacher_outputs2 = target_soft

					# print("teacher outputs after softmax >>>",F.softmax(teacher_outputs1))
					# print("Human soft outputs>>>",teacher_outputs2)
					# Knowledge Distillation Loss
					# loss_KD = nn.KLDivLoss()(F.log_softmax(output / T, dim=1),
													  # F.softmax(teacher_outputs / T, dim=1))
					# print(teacher_outputs.shape)
					# for x in range(teacher_outputs2.shape[0]):
					# 	# print(torch.sum(- teacher_outputs2 * F.log_softmax(teacher_outputs1, dim=1), dim=1)[x].data)
					# 	if torch.sum(- teacher_outputs2 * F.log_softmax(teacher_outputs1, dim=1), dim=1)[x].data >3.:
					# 	 	diverged_examples.append(x)
					# 	else:
					# 		non_diverged_examples.append(x)
					# print(diverged_examples)
					# # loss_KD= nn.KLDivLoss()(F.log_softmax(output / T, dim=1),
					# #  								 F.softmax(teacher_outputs / T, dim=1) )

					# loss_KD2= criterion1(F.softmax((output/T).index_select(0,torch.LongTensor(diverged_examples).cuda()),dim=1), teacher_outputs2.index_select(0,torch.LongTensor(diverged_examples).cuda()))

					# # loss_KD2= torch.mean(torch.sum((- 0.8* F.softmax(teacher_outputs1 /T , dim=1) -0.2*teacher_outputs2)* F.log_softmax(output/T, dim=1), dim=1).index_select(0,torch.LongTensor(diverged_examples).cuda()))

					# # print(torch.sum(- teacher_outputs2 * F.log_softmax(output, dim=1), dim=1).shape)
		
					# loss_KD1 = torch.mean(torch.sum(- F.softmax(teacher_outputs1 /T , dim=1) * F.log_softmax(output/T, dim=1), dim=1).index_select(0,torch.LongTensor(non_diverged_examples).cuda()))

					# loss_KD2= criterion1(F.softmax(output/T,dim=1), teacher_outputs2 )
					# # loss_KD2 = torch.mean(torch.sum(- teacher_outputs2 * F.log_softmax(output, dim=1), dim=1))
					loss_KD1 = torch.mean(torch.sum(- F.softmax(teacher_outputs1 /T , dim=1) * F.log_softmax(output/T, dim=1), dim=1))
					# print(loss_KD1,loss_SL)
					# loss_KD = nn.KLDivLoss()(F.log_softmax(output / T, dim=1),
					#  								  teacher_outputs)
					# loss_KD = criterion1(F.softmax(output / T,dim=1),teacher_outputs)
					# loss = (1 - lambda_) * loss_SL + lambda_ * T * T * loss_KD2
					# if torch.isnan(loss_KD2):
					# 	loss_KD = loss_KD1
					# else:
					# loss_KD = (len(non_diverged_examples)/self.config['batch_size']) * loss_KD1 + (len(diverged_examples)/self.config['batch_size'])* loss_KD2
					loss = gamma_*((1 - lambda_) * loss_SL + lambda_ * T * T * loss_KD1) # + (1-gamma_)*loss_KD2  #+ 0.3 * T * T * loss_KD2
					# total_loss+=loss
				loss_SL.backward()
				self.optimizer.step()
			# print("loss>>",total_loss.item())
			# training_losses.append(total_loss.item())
			
			print("epoch {}/{}".format(epoch, epochs))
			
			val_acc, val_loss = self.validate(step=epoch)
			validation_losses.append(val_loss)
			if val_acc > best_acc:
				best_acc = val_acc
				best_cefits = (total_loss.item() *self.config['batch_size'])/len(self.train_loader.dataset)
				# self.save(epoch, name='{}_{}_train10_test50_customdata_softlabels_teacherhumandiv_r14TA_besteacher_resnet39th_lambda{}_temp{}_best.pth.tar'.format(self.name, trial_id,lambda_,T))
				# self.save(epoch, name='{}_{}_train10_test50_customdata_softlabels_basekd_teacher_resnet110_{}th_lambda{}_temp{}_best.pth.tar'.format(self.name, trial_id,trial_id,lambda_,T))
				
				self.save(epoch, name='{}_{}_train10_test50_customdata_softlabels_teacheronly_resnet110_best.pth.tar'.format(self.name, trial_id))
		
		
		return best_acc, best_cefits
	
	def validate(self, step=0):
	
		self.student.eval()
		criterion = nn.CrossEntropyLoss()
		with torch.no_grad():
			correct = 0
			total = 0
			acc = 0
			total_val_loss=0
			for (images,labels,_) in self.test_loader:
				# print(images,labels)
				images = images.to(self.device)
				labels = labels.to(self.device)
				outputs = self.student(images)
				loss_val= criterion(outputs, labels) 
				total_val_loss+=loss_val.item()
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
			# self.accuracy_history.append(acc)
			acc = 100 * correct / total
			for param_group in self.optimizer.param_groups:
				print('Learning rate:' + str(param_group['lr']))
			self.scheduler.step(acc)
			print('{{"metric": "{}_val_accuracy", "value": {}}}'.format(self.name, acc))
			return acc, total_val_loss
	
	def save(self, epoch, name=None):
		trial_id = self.config['trial_id']
		if name is None:
			torch.save({
				'epoch': epoch,
				'model_state_dict': self.student.state_dict(),
				'optimizer_state_dict': self.optimizer.state_dict(),
			}, '/tigress/smondal/{}_{}_epoch{}.pth.tar'.format(self.name, trial_id, epoch))
		else:
			torch.save({
				'model_state_dict': self.student.state_dict(),
				'optimizer_state_dict': self.optimizer.state_dict(),
				'epoch': epoch,
			}, "/tigress/smondal/"+name)
	
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


if __name__ == "__main__":
	# Parsing arguments and prepare settings for training
	logs = open('/tigress/smondal/softlabels_50teachersonly(resnet110).log', 'a')
	args = parse_arguments()
	print(args)
	# config = nni.get_next_parameter()
	# config={'trial_id':np.arange(70).tolist(),'lambda_student':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95],
	# 'T_student':[1,2,4,5,8,10,20]}
	
	config={'trial_id':np.arange(5).tolist(),'lambda_student':[0.6],
	 'T_student':[17]}
	
	# torch.manual_seed(31)
	# torch.cuda.manual_seed(31)
	trial_id = os.environ.get('NNI_TRIAL_JOB_ID')
	dataset = args.dataset
	num_classes = 100 if dataset == 'cifar100' else 10
	teacher_model = None
	student_model = create_cnn_model(args.student, dataset, use_cuda=args.cuda)
	count=0
	highest_acc=0
	lambda_best=0
	temperature_best=0

	for i in range(1):
		for j in range(1):
	
			teacher_model = None
			student_model = create_cnn_model(args.student, dataset, use_cuda=args.cuda)
			train_config = {
				'epochs': args.epochs,
				'learning_rate': args.learning_rate,
				'momentum': args.momentum,
				'weight_decay': args.weight_decay,
				'device': 'cuda' if args.cuda else 'cpu',
				'is_plane': not is_resnet(args.student),
				'trial_id': args.trial_id,

				# 'T_student': config.get('T_student')[i],
				# 'lambda_student': config.get('lambda_student')[i],
				'batch_size': args.batch_size,
				'temperature': args.temperature,
				'lambda_': args.lambda_,
				'gamma_': args.gamma_
			}
			
			
			# Train Teacher if provided a teacher, otherwise it's a normal training using only cross entropy loss
			# This is for training single models(NOKD in paper) for baselines models (or training the first teacher)
			if args.teacher:
				teacher_model = create_cnn_model(args.teacher, dataset, use_cuda=args.cuda)
				if args.teacher_checkpoint:
					print("---------- Loading Teacher -------")
					teacher_model = load_checkpoint(teacher_model, args.teacher_checkpoint)
				else:
					print("---------- Training Teacher -------")
					train_loader, test_loader = get_cifar(num_classes,batch_size=args.batch_size,crop=True)
					teacher_train_config = copy.deepcopy(train_config)
					teacher_name = '{}_{}_best.pth.tar'.format(args.teacher, trial_id)
					teacher_train_config['name'] = args.teacher
					teacher_trainer = TrainManager(teacher_model, teacher=None, train_loader=train_loader, test_loader=test_loader, train_config=teacher_train_config)
					best_valacc, best_cefits = teacher_trainer.train()
					logline = str(train_config['trial_id']) + ','+ str(best_valacc) + ','+ str(best_cefits) +  '\n'
					# teacher_model = load_checkpoint(teacher_model, os.path.join('./', teacher_name))
					
			# Student training
			# print("---------- Training Student -------")
			# student_train_config = copy.deepcopy(train_config)
			# train_loader, test_loader = get_cifar(num_classes,batch_size=args.batch_size,crop=True)
			# student_train_config['name'] = args.student
			# student_trainer = TrainManager(student_model, teacher=teacher_model, train_loader=train_loader, test_loader=test_loader, train_config=student_train_config)
			# best_student_acc, validation_losses = student_trainer.train()
			# print("Best student accuarcy obtained for baseline KD with no = %s, gamma= %s, lambda= %s, temperature= %s is %s"%(train_config['trial_id'],train_config['gamma_'],train_config['lambda_'],train_config['temperature'],best_student_acc))
			# # with open("hkd3_validationlosses.txt", "wb") as fp:   #Pickling
			# # 	pickle.dump(validation_losses, fp)
			# if best_student_acc > highest_acc:
			# 	highest_acc = best_student_acc
			# 	lambda_best=train_config['lambda_']
			# 	temperature_best=train_config['temperature'] 
			# print("Overall Best student accuarcy obtained for baseline KD till now is with lambda= %s, temperature= %s is %s"%(lambda_best,temperature_best,highest_acc))
			# count+=1
			# logline = str(train_config['trial_id']) + ','+ str(best_student_acc) +  '\n'
			# nni.report_final_result(best_student_acc)

	print("Overall Best Validation accuracy for Teacher-only is %s, cross entropy with human softs is %s,"%(best_valacc,best_cefits))
	
	logs.write(logline)
	logs.flush()
	
	os.fsync(logs.fileno())
 No newline at end of file
iff --git a/train10k_images.npy b/train10k_images.npy
ew file mode 100644
ndex 0000000..7572a2b
inary files /dev/null and b/train10k_images.npy differ
iff --git a/train10k_labels.npy b/train10k_labels.npy
ew file mode 100644
ndex 0000000..e28ed34
inary files /dev/null and b/train10k_labels.npy differ
iff --git a/train_manager.py b/train_manager.py
ew file mode 100644
ndex 0000000..d51af46
-- /dev/null
++ b/train_manager.py
@ -0,0 +1,165 @@
import os
# import nni
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_loader import get_cifar
from model_factory import is_resnet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

def index_probabilities(teacherProbs, batch_idx):
	# is this right?
	return teacherProbs[batch_idx.to(self.device).long()]

def prepare_teacher_targets(teacherProbs, batch_idx):
	# index batch
	probsBatch = index_probabilities(teacherProbs, batch_idx)
	return probsBatch

class TrainManager(object):
	def __init__(self, student, teacherProbs=None, train_loader=None, 
		test_loader=None, train_config={}):
		self.config = train_config
		self.student = student
		self.teacherProbs = teacherProbs
		self.train_loader = train_loader
		self.test_loader = test_loader
		
		self.optimizer = optim.SGD(self.learner.parameters(),
								   lr=self.config['learning_rate'],
								   momentum=self.config['momentum'],
								   weight_decay=self.config['weight_decay'])

		self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', factor = 0.9, patience = 5, 
			verbose = True)
			

	def train(self):
		epochs = self.config['epochs']
		trial_id = self.config['trial_id']
		lambda_ = self.config['lambda_']
		T = self.config['temperature']

		iteration = 0
		best_acc = 0


		validation_losses=[]
		training_losses=[]

		criterion = nn.CrossEntropyLoss()
		distillation_criterion = nn.KLDivLoss()

		print('Starting student training, no = {} >>>>>>>>>>>>>'.format(trial_id))

		for epoch in range(epochs):
			self.student.train()
			for batch_idx, (data, target_hard, target_soft) in enumerate(self.train_loader):
				
				iteration += 1

				total_loss = 0
				total_loss_SL = 0
				total_loss_KD = 0

                self.optimizer.zero_grad()

				data = data.to(self.device).float()
				target_hard = target_hard.to(self.device) #.long() # may need this to survive
				output = self.student(data)

				# data loss
				loss_SL = criterion(output, target_hard)
			    
				# Knowledge Distillation loss
				# return teacher targets by indexing into teacherProbs with batch_idx
				if self.teacherProbs:
                    teacher_outputs = prepare_teacher_targets(self.teacherProbs, 
														  batch_idx)
				else:
				    teacher_outputs = target_soft
				loss_KD = distillation_criterion(F.log_softmax(output / T, dim=1),
											     F.softmax(teacher_outputs / T, dim=1))

				# total loss
				loss = (1 - lambda_) * loss_SL + lambda_ * T * T * loss_KD
				total_loss += loss
				total_loss_SL += loss_SL
				total_loss_KD += loss_KD
				loss.backward()
				self.optimizer.step()

			print("training loss epoch>>",total_loss)
			training_losses.append((total_loss, total_loss_SL, total_loss_KD))
			
			print("epoch {}/{} done".format(epoch, epochs))
			
			val_acc, val_loss = self.validate(step=epoch)
			validation_losses.append((val_acc, val_loss))

			if val_acc > best_acc:
				best_acc = val_acc
				best_loss = val_loss
				self.save(epoch, name = self.config['outfile'], training_losses, validation_losses)


			#training and validation losses aren't used, yet
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
iff --git a/train_student.py b/train_student.py
ew file mode 100644
ndex 0000000..cd7bb31
-- /dev/null
++ b/train_student.py
@ -0,0 +1,120 @@
import os
# import nni
import copy
import pickle
import torch
import argparse
from data_loader import get_cifar
from find_best_teacher import load_best_model

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
	parser.add_argument('--master_architecture', default='', type=str, help='next level down from model dump')
	parser.add_argument('--teacher', default='human', type=str, help='teacher name')
	parser.add_argument('--student', default='', type=str, help='student name')
	parser.add_argument('--manual_seed', default=0, type=int, help='manual seed')
	parser.add_argument('--iter', default=0, type=int, help='manual seed')

	parser.add_argument('--temperature', default=1, type=int,  help='student temperature')
	parser.add_argument('--lambda_', default=0.5, type=float,  help='weighted average')
	parser.add_argument('--gamma_', default=0.5, type=float,  help='weighted average')

	parser.add_argument('--epochs', default=500, type=int,  help='number of total epochs to run')
	parser.add_argument('--dataset', default='cifar10', type=str, help='dataset. can be either cifar10 or cifar100')
	parser.add_argument('--batch-size', default=128, type=int, help='batch_size')
	parser.add_argument('--learning-rate', default=0.1, type=float, help='initial learning rate')
	parser.add_argument('--momentum', default=0.9, type=float,  help='SGD momentum')
	parser.add_argument('--weight-decay', default=1e-4, type=float, help='SGD weight decay (default: 1e-4)')
	
	parser.add_argument('--cuda', default=False, type=str2bool, help='whether or not use cuda(train on GPU)')
	parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')
	parser.add_argument('--trial_id', default=1, type=int,  help='id number')
	
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	# Parsing arguments and prepare settings for training
	student_dir = '{}/teacher_{}'
	args = parse_arguments()
	print(args)
    log_path = '{}/{}/{}/training_log.log'.format(args.master_outdir, args.master_architecture, args.teacher)
    save_path = '{}/{}/{}/seed_{}'.format(args.master_outdir, args.master_architecture, args.teacher, args.manual_seed)

    if not os.path.exists():
    	print('making new dirs')
    	os.makedirs(save_path)

	logs = open(log_path, 'a')

	torch.manual_seed(args.manual_seed)
	torch.cuda.manual_seed(args.manual_seed)
	trial_id = os.environ.get('NNI_TRIAL_JOB_ID')
	dataset = args.dataset

	if dataset == 'cifar10':
		num_classes = 10 
	else:
		print('cifar10 not loaded!')

	train_config = {
				'epochs': args.epochs,
				'learning_rate': args.learning_rate,
				'momentum': args.momentum,
				'weight_decay': args.weight_decay,
				'device': 'cuda' if args.cuda else 'cpu',
				'trial_id': args.trial_id,
				'batch_size': args.batch_size,
				'teacher': args.teacher,
				'lambda_': args.lambda_,
				'temperature': args.temperature,
				'gamma_': args.gamma_
			}

	student_model = create_cnn_model(args.student, dataset, use_cuda=args.cuda)

	# below will be dict with name and probs
	teacher_model = load_best_model(args.teacher, args.master_outdir)

	train_loader, test_loader = get_cifar(num_classes, batch_size=args.batch_size, crop=True)

	student_name = 'student_{}_temperature_{}_lambda_{}_gamma_{}_trial_{}_best.pth.tar'.format(args.student, 
								args.temperature, args.lambda_, args.gamma_, trial_id)
	train_config['outfile'] = '{}/{}'.format(save_path, student_name)

	print("---------- Training Student -------")
	
	student_trainer = TrainManager(student_model, teacherProbs=teacher_model['probs'], 
		                           train_loader=train_loader, test_loader=test_loader, train_config=train_config)

	best_valacc, best_valloss = student_trainer.train()

	print("Best student accuacy for teacher {} student {}, trial = {}, gamma= {}, lambda= {}, temperature= {} is {}".format(args.teacher,
		args.student, args.trial_id, args.gamma_, args.lambda_, args.temperature, best_valacc))

	logline = "{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, \n".format(args.teacher,
		      args.student, args.trial_id, args.gamma_, args.lambda_, args.temperature, best_valacc, best_valloss)
				
	logs.write(logline)
	logs.flush()
	
	os.fsync(logs.fileno())
 No newline at end of file
iff --git a/wrn.py b/wrn.py
ew file mode 100644
ndex 0000000..90b501d
-- /dev/null
++ b/wrn.py
@ -0,0 +1,172 @@
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_in')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.uniform_()
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, drop_rate):
        super(BasicBlock, self).__init__()

        self.drop_rate = drop_rate

        self._preactivate_both = (in_channels != out_channels)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,  # downsample with first conv
            padding=1,
            bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))

    def forward(self, x):
        if self._preactivate_both:
            x = F.relu(
                self.bn1(x), inplace=True)  # shortcut after preactivation
            y = self.conv1(x)
        else:
            y = F.relu(
                self.bn1(x),
                inplace=True)  # preactivation only for residual path
            y = self.conv1(y)
        if self.drop_rate > 0:
            y = F.dropout(
                y, p=self.drop_rate, training=self.training, inplace=False)

        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y += self.shortcut(x)
        return y


class Network3(nn.Module):
    def __init__(self):
        super(Network3, self).__init__()

        input_shape = (1,3,32,32)
        n_classes = 10

        base_channels = 16
        widening_factor = 10
        drop_rate = 0
        depth = 28

        block = BasicBlock
        n_blocks_per_stage = (depth - 4) // 6
        assert n_blocks_per_stage * 6 + 4 == depth

        n_channels = [
            base_channels,
            base_channels * widening_factor,
            base_channels * 2 * widening_factor,
            base_channels * 4 * widening_factor
        ]

        self.conv = nn.Conv2d(
            input_shape[1],
            n_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        self.stage1 = self._make_stage(
            n_channels[0],
            n_channels[1],
            n_blocks_per_stage,
            block,
            stride=1,
            drop_rate=drop_rate)
        self.stage2 = self._make_stage(
            n_channels[1],
            n_channels[2],
            n_blocks_per_stage,
            block,
            stride=2,
            drop_rate=drop_rate)
        self.stage3 = self._make_stage(
            n_channels[2],
            n_channels[3],
            n_blocks_per_stage,
            block,
            stride=2,
            drop_rate=drop_rate)
        self.bn = nn.BatchNorm2d(n_channels[3])

        # compute conv feature size
        with torch.no_grad():
            self.feature_size = self._forward_conv(
                torch.zeros(*input_shape)).view(-1).shape[0]

        self.fc = nn.Linear(self.feature_size, n_classes)

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride,
                    drop_rate):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                stage.add_module(block_name,
                                 block(
                                     in_channels,
                                     out_channels,
                                     stride=stride,
                                     drop_rate=drop_rate))
            else:
                stage.add_module(block_name,
                                 block(
                                     out_channels,
                                     out_channels,
                                     stride=1,
                                     drop_rate=drop_rate))
        return stage

    def _forward_conv(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
 No newline at end of file
