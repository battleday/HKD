import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
NUM_WORKERS = 32
import numpy as np

from PIL import Image
class cifar_softhard(Dataset):

	def __init__(self,train=True):
		normalize=transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
		# normalize1=transforms.Normalize(mean = [0.47889522, 0.47227842, 0.43047404],std = [0.24205776, 0.23828046, 0.25874835])
		if train:
			file1 = './data/train10k_images.npy'
			file2= './data/train10k_labels.npy'
			# file3= './data/cifar10h-probs.npy'
			self.transform = transforms.Compose([transforms.ToPILImage(),
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize
		])
		else:
			# file1 = 'test50k_images.npy'
			# file2 ='test50k_labels.npy'
			# file3= 'test50k_labels.npy'
			# self.transform=transforms.Compose([transforms.ToPILImage(),
			# 	transforms.ToTensor(),
			# 	normalize])


			file1 = '/tigress/ruairidh/HKD/imagenet_far_data/test_imagenetfar_images.npy'
			file2 ='/tigress/ruairidh/HKD/imagenet_far_data/test_imagenetfar_labels.npy'
			# file3= '/tigress/ruairidh/HKD/imagenet_far_data/test_imagenetfar_labels.npy'
			self.transform=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()
				,normalize])
			
		self.images_root = np.load(file1)
		self.labels_hard= np.load(file2)
		# self.labels_soft= np.load(file3)

		if self.images_root.max()==255.:

			self.images = [self.transform(torch.from_numpy((x.transpose((2,0,1)).astype(np.uint8)))) for x in self.images_root]
		else:
			self.images = [self.transform(torch.from_numpy((x*255).astype(np.uint8))) for x in self.images_root]

		
	#	files = os.listdir(self.images_root)
		
		
#        self.filenames = [f[:-4] for f in files]
#        self.filenames=sorted(files)

	#	self.input_transform = input_transform
	#	self.target_transform = target_transform

	def __getitem__(self, index):
		
		# print(self.images_root[index].shape)
		# self.images_root[index] = (self.images_root[index] + 1) * 127.5
		# if self.images_root.max()==255.:

		# 	image = self.transform(torch.from_numpy((self.images_root[index].transpose((2,0,1)).astype(np.uint8))))
		# else:

		# 	image = self.transform(torch.from_numpy((self.images_root[index]*255).astype(np.uint8)))
		image = self.images[index]
		hard= self.labels_hard[index]
		# soft = self.labels_soft[index]
		
		#print('before>>>',filename)
		
		#print('after>>>',filename)
		
	   

# 		with open(image_path(self.images_root, filename), 'rb') as f:
# 			image =
		# print(np.max(self.images_root[index]),np.min(self.images_root[index]))
		
		return image, hard

	def __len__(self):
		return len(self.images_root)







def get_imagenet(num_classes=100, dataset_dir='./data', batch_size=1, crop=False):
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
	
	# if num_classes == 100:
	# 	trainset = torchvision.datasets.CIFAR100(root=dataset_dir, train=True,
	# 											 download=True, transform=train_transform)
		
	# 	testset = torchvision.datasets.CIFAR100(root=dataset_dir, train=False,
	# 											download=True, transform=simple_transform)
	# else:
	# 	trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True,
	# 											 download=True, transform=simplest_transform)
		
	# 	testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
	# 											download=True, transform=simplest_transform)
		
	# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=NUM_WORKERS,
	# 										  pin_memory=True, shuffle=False)
	# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=NUM_WORKERS,
	# 										 pin_memory=True, shuffle=False)

	
	trainloader = torch.utils.data.DataLoader(cifar_softhard(train=True), batch_size=batch_size,
											  pin_memory=True,shuffle=True, num_workers=NUM_WORKERS) # Creating dataloader

	#testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=apply_transform)
	
	testloader = torch.utils.data.DataLoader(cifar_softhard(train=False), batch_size=batch_size,pin_memory=True,
	 										 shuffle=False, num_workers=NUM_WORKERS) # Creating dataloader
	
	
	# cinic_mean = [0.47889522, 0.47227842, 0.43047404]
	# cinic_std = [0.24205776, 0.23828046, 0.25874835]
	# testloader= torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(cinic_directory + 'all_combined/',
 #    	transform=transforms.Compose([transforms.ToTensor(),
 #        transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
 #    batch_size=batch_size, shuffle=False)
	print('No. of samples in train set: '+str(len(trainloader.dataset)))
	print('No. of samples in test set: '+str(len(testloader.dataset)))


	return trainloader, testloader


if __name__ == "__main__":
	# print("CIFAR10")
	# trainloader, testloader= get_cifar(10,crop=True)
	# print("---"*20)
	# print("---"*20)
	# print("CIFAR100")
	# # print(get_cifar(100))
	# test_data= torch.Tensor().cuda().float()
	# test_target= torch.Tensor().cuda().long()
	# for batch_idx, (data, target) in enumerate(testloader):
	# 	test_data=torch.cat([test_data,data.cuda()],dim=0)
	# 	test_target=torch.cat([test_target,target.cuda()],dim=0)
	# 	print(batch_idx)

	# train_data= torch.Tensor().cuda().float()
	# train_target= torch.Tensor().cuda().long()
	# for batch_idx, (data, target) in enumerate(trainloader):
	# 	train_data=torch.cat([train_data,data.cuda()],dim=0)
	# 	train_target=torch.cat([train_target,target.cuda()],dim=0)
	# 	print(batch_idx)
		
	# print('Shape of data and targets are >>>',test_data.cpu().numpy().shape,test_target.cpu().numpy() )
	# print('first pil test image>>>',Image.fromarray((test_data.cpu().numpy()[0]*255).astype(np.uint8).reshape(32,32,3)))


	# np.save('train10k_images.npy',test_data.cpu().numpy())
	# np.save('train10k_labels.npy',test_target.cpu().numpy())
	# np.save('test50k_images.npy',train_data.cpu().numpy())


	classes= ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
	# np.save('test50k_labels.npy',train_target.cpu().numpy())
	
	# cinic_mean = [0.47889522, 0.47227842, 0.43047404]
	# cinic_std = [0.24205776, 0.23828046, 0.25874835]
	# test_data= torch.Tensor().cuda().float()
	# count=0

	# testloader= torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root=cinic_directory + 'all_combined/',
	# transform=transforms.Compose([transforms.ToTensor(),
 #    transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
	# batch_size=128, shuffle=False)
	
	# test_target= torch.Tensor().cuda().long()
	# for batch_idx, (data, target) in enumerate(testloader):
	# 	test_data=torch.cat([test_data,data.cuda()],dim=0)
	# 	test_target=torch.cat([test_target,target.cuda()],dim=0)
	# 	print(batch_idx)
		
	
	# np.save('test210k_cinic10_imagenet_images.npy',test_data.cpu().numpy())
	# np.save('test210k_cinic10_imagenet_labels.npy',test_target.cpu().numpy())
	images=np.empty([1,32,32,3])
	labels=np.zeros(63895)
	count=0
	for idx,c in enumerate(classes):
		z=np.load('imagenet_replacements/'+c+'.npy').shape[0]
		labels[count:count+z] =idx
		images=np.concatenate([images,np.load('imagenet_replacements/'+c+'.npy')],axis=0)
		count=count+z
		print(c)
	
	print(images,labels)
	np.save('test_imagenetfar_images.npy',images[1:])
	np.save('test_imagenetfar_labels.npy',labels)
	# np.save('test50k_images.npy',train_data.cpu().numpy())

