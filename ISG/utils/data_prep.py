import os
import sys
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
from pathlib import Path
from matplotlib import pyplot as plt



def load_datasets(args, task_num):
	if args.model_type == 'CNN':
		#Load datasets
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		# Normalize the test set same as training set without augmentation
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		if task_num == -1:  #CIFAR10 
			trainset = torchvision.datasets.CIFAR10(root="../../data", train=True, download=False, transform=transform_train)
			testset = torchvision.datasets.CIFAR10(root="../../data", train=False, download=False, transform=transform_test)

		else:				#CIFAR100 
			trainset = torchvision.datasets.CIFAR100(root="../../data", train=True, download=False, transform=transform_train)
			testset = torchvision.datasets.CIFAR100(root="../../data", train=False, download=False, transform=transform_test)

			#split the dataset
			trainset = split_CIFAR100_dataset(trainset,task_num)
			testset  = split_CIFAR100_dataset(testset, task_num)

		#dataset loaders
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=2)
		testloader  = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=2)
		# maskloader  = torch.utils.data.DataLoader(testset, batch_size = 1000, shuffle = True, num_workers = 2)


	elif args.model_type == 'MLP':
		if args.padding:
			#Transform with padding
			transform = transforms.Compose([
				transforms.Pad(2, fill=0, padding_mode='constant'),
				transforms.ToTensor(),
				transforms.Normalize(mean=(0.1000,), std=(0.2752,)), # for 32x32
			])
		else:
			#Transforms
			transform = transforms.Compose(
				[transforms.ToTensor(),
				transforms.Normalize(mean=(0.1307,), std=(0.3081,))]) # for 28x28

		#Datasets
		trainset = torchvision.datasets.MNIST('../../data',
			download = True,
			train = True,
			transform = transform)
		testset = torchvision.datasets.MNIST('../../data',
			download = True,
			train = False,
			transform = transform)

	#Dataloader
	trainloader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size_train, shuffle = True, num_workers = 2)
	testloader = torch.utils.data.DataLoader(testset, batch_size = args.batch_size_test, shuffle = False, num_workers = 2)
	# maskloader = torch.utils.data.DataLoader(testset, batch_size = args.batch_size_fisher, shuffle = True, num_workers = 2)

	return trainloader, testloader



def permute_MNIST(data, shuffle_idx, task_num, args):
	if args.padding:
		data = torch.stack([x.view(-1)[shuffle_idx[task_num]].view(1, 32, 32) for x in data], dim=0)
	else:
		data = torch.stack([x.view(-1)[shuffle_idx[task_num]].view(1, 28, 28) for x in data], dim=0)
	return data

#pMNIST shuffle
def pMNIST_shuffle(args):
	#list of shuffling index for pMNIST
	shuffle_idx = []
	if args.padding:
		shuffle_idx.append(torch.arange(32*32)) #First take is the original
		for t in range(args.num_task-1):
			shuffle_idx.append(torch.randperm(32*32))
	else:
		shuffle_idx.append(torch.arange(28*28)) #First take is the original
		for t in range(args.num_task-1):
			shuffle_idx.append(torch.randperm(28*28))
	return shuffle_idx

def split_CIFAR100_dataset(dataset, task_num):
	'''
	CIFAR10 trainset
		-Compose()
		-train: True
		-data: (50000, 32, 32, 3) <--- np.ndarray
		-targets: 50000 label integers <--- list 
		-classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
		-class_to_idx: {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
	CIFAR100 trainset
		-data: (50000, 32, 32, 3)
		-classes: ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
		-class_to_idx: {'apple': 0, 'aquarium_fish': 1, 'baby': 2, 'bear': 3, 'beaver': 4, 'bed': 5, 'bee': 6, 'beetle': 7, 'bicycle': 8, 'bottle': 9, 'bowl': 10, 'boy': 11, 'bridge': 12, 'bus': 13, 'butterfly': 14, 'camel': 15, 'can': 16, 'castle': 17, 'caterpillar': 18, 'cattle': 19, 'chair': 20, 'chimpanzee': 21, 'clock': 22, 'cloud': 23, 'cockroach': 24, 'couch': 25, 'crab': 26, 'crocodile': 27, 'cup': 28, 'dinosaur': 29, 'dolphin': 30, 'elephant': 31, 'flatfish': 32, 'forest': 33, 'fox': 34, 'girl': 35, 'hamster': 36, 'house': 37, 'kangaroo': 38, 'keyboard': 39, 'lamp': 40, 'lawn_mower': 41, 'leopard': 42, 'lion': 43, 'lizard': 44, 'lobster': 45, 'man': 46, 'maple_tree': 47, 'motorcycle': 48, 'mountain': 49, 'mouse': 50, 'mushroom': 51, 'oak_tree': 52, 'orange': 53, 'orchid': 54, 'otter': 55, 'palm_tree': 56, 'pear': 57, 'pickup_truck': 58, 'pine_tree': 59, 'plain': 60, 'plate': 61, 'poppy': 62, 'porcupine': 63, 'possum': 64, 'rabbit': 65, 'raccoon': 66, 'ray': 67, 'road': 68, 'rocket': 69, 'rose': 70, 'sea': 71, 'seal': 72, 'shark': 73, 'shrew': 74, 'skunk': 75, 'skyscraper': 76, 'snail': 77, 'snake': 78, 'spider': 79, 'squirrel': 80, 'streetcar': 81, 'sunflower': 82, 'sweet_pepper': 83, 'table': 84, 'tank': 85, 'telephone': 86, 'television': 87, 'tiger': 88, 'tractor': 89, 'train': 90, 'trout': 91, 'tulip': 92, 'turtle': 93, 'wardrobe': 94, 'whale': 95, 'willow_tree': 96, 'wolf': 97, 'woman': 98, 'worm': 99}
	CIFAR100 testset
	    -data: (10000, 32, 32, 3)
	'''
	targets = []
	data = []
	class_idx = range(10*task_num, 10*task_num+10)
	dataset.classes = dataset.classes[10*task_num: 10*(task_num+1)]

	class_to_idx = {}
	for i, x in enumerate(dataset.classes):
		class_to_idx[x] = i
	dataset.class_to_idx = class_to_idx

	for i, x in enumerate(dataset.targets):
		if x in class_idx:
			targets.append(x-10*task_num)
			data.append(dataset.data[i])

	dataset.targets = targets
	dataset.data = data
	# plot_image(dataset)
	return dataset

def plot_image(dataset):
	for i in range(9):
		target =np.array(dataset.targets)
		# print(np.where(target == 1)[0][0])
		plt.subplot(3, 3, i+1)
		plt.imshow(dataset.data[np.where(target == i)[0][0]])
	plt.show()

def check_split():
	for task_num in range(10):
		dataset = torchvision.datasets.CIFAR100(root="../../data", train=True, download=True)
		dset = split_CIFAR100_dataset(dataset, task_num)
		print(dataset.classes)
		plot_image(dataset)