
+++ b/data_loader.py
@@ -0,0 +1,160 @@
+import torch
+import torchvision
+import torchvision.transforms as transforms
+from torch.utils.data import Dataset
+NUM_WORKERS = 32
+import numpy as np
+
+from PIL import Image
+class cifar_softhard(Dataset):
+
+	def __init__(self,train=True):
+		normalize=transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
+
+		if train:
+			file1 = 'train10k_images.npy'
+			file2= 'train10k_labels.npy'
+			file3= 'cifar10h-probs.npy'
+			self.transform = transforms.Compose([transforms.ToPILImage(),
+			transforms.RandomCrop(32, padding=4),
+			transforms.RandomHorizontalFlip(),
+			transforms.ToTensor(),
+			normalize
+		])
+		else:
+			file1 = 'test50k_images.npy'
+			file2 ='test50k_labels.npy'
+			file3= 'test50k_labels.npy'
+			self.transform=transforms.Compose([transforms.ToPILImage(),
+				transforms.ToTensor(),
+				normalize])
+			
+		self.images_root = np.load(file1)
+		self.labels_hard= np.load(file2)
+		self.labels_soft= np.load(file3)
+
+	#	files = os.listdir(self.images_root)
+		
+		
+#        self.filenames = [f[:-4] for f in files]
+#        self.filenames=sorted(files)
+
+	#	self.input_transform = input_transform
+	#	self.target_transform = target_transform
+
+	def __getitem__(self, index):
+		
+		# print(self.images_root[index].shape)
+		# self.images_root[index] = (self.images_root[index] + 1) * 127.5
+		
+		image = self.transform(torch.from_numpy((self.images_root[index]*255).astype(np.uint8)))
+		hard= self.labels_hard[index]
+		soft = self.labels_soft[index]
+		
+		#print('before>>>',filename)
+		
+		#print('after>>>',filename)
+		
+	   
+
+# 		with open(image_path(self.images_root, filename), 'rb') as f:
+# 			image =
+		
+		return image, hard, soft
+
+	def __len__(self):
+		return len(self.images_root)
+
+
+
+
+
+
+
+def get_cifar(num_classes=100, dataset_dir='./data', batch_size=1, crop=False):
+	"""
+	:param num_classes: 10 for cifar10, 100 for cifar100
+	:param dataset_dir: location of datasets, default is a directory named 'data'
+	:param batch_size: batchsize, default to 128
+	:param crop: whether or not use randomized horizontal crop, default to False
+	:return:
+	"""
+	normalize=transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
+	# normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
+	simple_transform = transforms.Compose([transforms.ToTensor(), normalize])
+	
+	simplest_transform = transforms.Compose([transforms.ToTensor()])
+	
+
+	if crop is True:
+		train_transform = transforms.Compose([
+			transforms.RandomCrop(32, padding=4),
+			transforms.RandomHorizontalFlip(),
+			transforms.ToTensor(),
+			normalize
+		])
+	else:
+		train_transform = simple_transform
+	
+	# if num_classes == 100:
+	# 	trainset = torchvision.datasets.CIFAR100(root=dataset_dir, train=True,
+	# 											 download=True, transform=train_transform)
+		
+	# 	testset = torchvision.datasets.CIFAR100(root=dataset_dir, train=False,
+	# 											download=True, transform=simple_transform)
+	# else:
+	# 	trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True,
+	# 											 download=True, transform=simplest_transform)
+		
+	# 	testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
+	# 											download=True, transform=simplest_transform)
+		
+	# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=NUM_WORKERS,
+	# 										  pin_memory=True, shuffle=False)
+	# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=NUM_WORKERS,
+	# 										 pin_memory=True, shuffle=False)
+
+	
+	trainloader = torch.utils.data.DataLoader(cifar_softhard(train=True), batch_size=batch_size,
+											  pin_memory=True,shuffle=True, num_workers=NUM_WORKERS) # Creating dataloader
+
+	#testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=apply_transform)
+	testloader = torch.utils.data.DataLoader(cifar_softhard(train=False), batch_size=batch_size,pin_memory=True,
+											 shuffle=False, num_workers=NUM_WORKERS) # Creating dataloader
+	
+	print('No. of samples in train set: '+str(len(trainloader.dataset)))
+	print('No. of samples in test set: '+str(len(testloader.dataset)))
+
+
+	return trainloader, testloader
+
+
+if __name__ == "__main__":
+	print("CIFAR10")
+	trainloader, testloader= get_cifar(10,crop=True)
+	print("---"*20)
+	print("---"*20)
+	print("CIFAR100")
+	# print(get_cifar(100))
+	test_data= torch.Tensor().cuda().float()
+	test_target= torch.Tensor().cuda().long()
+	for batch_idx, (data, target) in enumerate(testloader):
+		test_data=torch.cat([test_data,data.cuda()],dim=0)
+		test_target=torch.cat([test_target,target.cuda()],dim=0)
+		print(batch_idx)
+
+	train_data= torch.Tensor().cuda().float()
+	train_target= torch.Tensor().cuda().long()
+	for batch_idx, (data, target) in enumerate(trainloader):
+		train_data=torch.cat([train_data,data.cuda()],dim=0)
+		train_target=torch.cat([train_target,target.cuda()],dim=0)
+		print(batch_idx)
+		
+	print('Shape of data and targets are >>>',test_data.cpu().numpy().shape,test_target.cpu().numpy() )
+	print('first pil test image>>>',Image.fromarray((test_data.cpu().numpy()[0]*255).astype(np.uint8).reshape(32,32,3)))
+
+
+	np.save('train10k_images.npy',test_data.cpu().numpy())
+	np.save('train10k_labels.npy',test_target.cpu().numpy())
+	np.save('test50k_images.npy',train_data.cpu().numpy())
+	np.save('test50k_labels.npy',train_target.cpu().numpy())