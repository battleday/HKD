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
