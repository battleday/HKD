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
	parser.add_argument('--iter', default=0, type=int, help='run')

	parser.add_argument('--temperature', default=1, type=int,  help='student temperature')
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
	parser.add_argument('--trial_id', default=1, type=int,  help='id number')
	
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	# Parsing arguments and prepare settings for training
	student_dir = '{}/teacher_{}'
	args = parse_arguments()
	print(args)
    log_path = '{}/{}/{}/seed_{}/training_log.log'.format(args.master_outdir, args.master_architecture, args.teacher, args.manual_seed)
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
                'disil_fn': args.distil_fn,
				'lambda_': args.lambda_,
				'temperature': args.temperature,
				'gamma_': args.gamma_
			}

	student_model = create_cnn_model(args.student, dataset, use_cuda=args.cuda)

	# below will be dict with name and probs
	teacher_model = load_best_model(args.teacher, args.master_outdir)

	train_loader, test_loader = get_cifar(num_classes, batch_size=args.batch_size, crop=True)

	student_name = 'student_{0}_distil_fn_{1}_temperature_{2}_lambda_{3}_gamma_{4}_iter_{5}_best.pth.tar'.format(args.student, 
								args.distil_fn, args.temperature, args.lambda_, args.gamma_, args.iter)
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
