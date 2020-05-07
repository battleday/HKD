import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
NUM_WORKERS = 32
import numpy as np
from PIL import Image
cinic_directory='/tigress/ruairidh/HKD/cinic_data/'

class cifar_softhard(Dataset):

	def __init__(self,train=True):
		normalize=transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
		if train:
			file1 = './data/train10k_images.npy'
			file2= './data/train10k_labels.npy'
			file3= './data/cifar10h-probs.npy'
			self.transform = transforms.Compose([transforms.ToPILImage(),
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize
		])
		else:
			file1 = './data/test50k_images.npy'
			file2 ='./data/test50k_labels.npy'
			file3= './data/test50k_labels.npy'
			self.transform=transforms.Compose([transforms.ToPILImage(),
				transforms.ToTensor(),
				normalize])


		self.images_root = np.load(file1)
		self.labels_hard= np.load(file2)
		self.labels_soft= np.load(file3)


	def __getitem__(self, index):
		
		if self.images_root.max()==255.:

			image = self.transform(torch.from_numpy((self.images_root[index].transpose((2,0,1)).astype(np.uint8))))
		else:

			image = self.transform(torch.from_numpy((self.images_root[index]*255).astype(np.uint8)))
		hard= self.labels_hard[index]
		soft = self.labels_soft[index]
		
		
		return image, hard, soft

	def __len__(self):
		return len(self.images_root)

def get_cinic(num_classes=100, dataset_dir='./data', batch_size=1, crop=False):
	"""
	:param num_classes: 10 for cifar10, 100 for cifar100
	:param dataset_dir: location of datasets, default is a directory named 'data'
	:param batch_size: batchsize, default to 128
	:param crop: whether or not use randomized horizontal crop, default to False
	:return:
	"""

	normalize=transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
	# normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
	simple_transform = transforms.Compose([transforms.ToTensor(), normalize])
	
	simplest_transform = transforms.Compose([transforms.ToTensor()])
	

	if crop is True:
		train_transform = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize
		])
	else:
		train_transform = simple_transform

	trainloader = torch.utils.data.DataLoader(cifar_softhard(train=True), batch_size=batch_size,
			pin_memory=True,shuffle=True, num_workers=NUM_WORKERS)
	cinic_mean = [0.47889522, 0.47227842, 0.43047404]
	cinic_std = [0.24205776, 0.23828046, 0.25874835]
	testloader= torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(cinic_directory + 'all_combined/',
    	transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
    batch_size=batch_size, shuffle=False)
	print('No. of samples in train set: '+str(len(trainloader.dataset)))
	print('No. of samples in test set: '+str(len(testloader.dataset)))

	return trainloader, testloader


