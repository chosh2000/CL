import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.optimizer_utils import *
from utils.train_utils import *
from utils.reg_utils import *
from utils.model_utils import *
from utils.data_prep import *


#Parameters
num_task = 10
n_epochs = 4
batch_size_train = 32
batch_size_test = 1000
learning_rate = 0.01
log_interval = 10
reg_lambda = 0.01 #default = 0.01
use_gpu = True
MLP_size = 2000
connect_ratio = 0.2
random_seed = 1
torch.manual_seed(random_seed)
multi_head = True
torch.backends.cudnn.enabled = True
method = "MAS" #MAS, EWC, SI
# momentum = 0.5

if method == "MAS":
	save_path = "pMNIST_MAS_nTask{}_epoch{}_mhead{}_reglambda{}_MLP{}_topk{}_biasdropped_newhead".format(str(num_task), str(n_epochs), str(multi_head),str(reg_lambda), str(MLP_size), str(connect_ratio))
if method == "EWC":
	save_path = "pMNIST_EWC_nTask{}_epoch{}_mhead{}_reglambda{}".format(str(num_task), str(n_epochs), str(multi_head), str(reg_lambda))
if method == "SI":
	save_path = "pMNIST_SI_nTask{}_epoch{}_mhead{}_reglambda{}".format(str(num_task), str(n_epochs), str(multi_head), str(reg_lambda))



#Transforms
transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5,), (0.5,))])

#Datasets
trainset = torchvision.datasets.MNIST('./data',
	download = True,
	train = True,
	transform = transform)
testset = torchvision.datasets.MNIST('./data',
	download = True,
	train = False,
	transform = transform)

#Dataloader
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size_train, shuffle = True, num_workers = 2)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size_test, shuffle = False, num_workers = 2)
maskloader = torch.utils.data.DataLoader(testset, batch_size = 10000, shuffle = True, num_workers = 2)


def mat_imshow(img, one_channel =False):
	if one_channel:
		img = img.mean(dim=0)
	# img = img / 2 + 0.5
	npimg = img.numpy()

	if one_channel:
		plt.imshow(npimg, cmap="Greys")
	else:
		plt.imshow(np.transpose(npimg, (1,2,0)))


class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv1 = nn.Conv2d(1,10,5)
		self.conv2 = nn.Conv2d(10,20,5)
		# self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(20*4*4, 50)
		self.fc_head = nn.Linear(50, 10)
	def forward(self, x): # x=28x28, batch=64
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = self.fc_head(x)
		return F.log_softmax(x)


class FcNet(nn.Module):
	def __init__(self):
		super(FcNet, self).__init__()
		self.fc1 = nn.Linear(28*28, MLP_size)
		self.fc2 = nn.Linear(MLP_size, MLP_size)
		self.fc_head = nn.Linear(MLP_size, 10)
	def forward(self, x):
		x = x.view(x.size(0), -1)  #convert tensor [batch_size,1,28,28] --> [batch_size,1*28*28]
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc_head(x)
		return x
		# return F.log_softmax(x)


def ISG_train():
	shuffle_idx = pMNIST_shuffle(num_task)
	for task_num in range(num_task):
		#Initialize model
		model = FcNet()
		# model = ConvNet()
		network = shared_model(model)
		network = model_init(model=network, task_num=task_num, num_classes=10, multi_head=multi_head, use_gpu = use_gpu)
		if task_num == 0:
			network = init_reg_params(network, use_gpu = use_gpu, freeze_layers = [])
		else:
			network = init_reg_params_across_tasks(network, multi_head = multi_head, use_gpu = use_gpu, freeze_layers = []) 
		# optimizer = optim.SGD(network.tmodel.parameters(), lr = learning_rate, momentum = momentum)


		# print_mask(network, [0,7])

		#Training
		if method == "MAS":
			train_MAS(network, task_num=task_num, n_epochs =n_epochs, trainloader=trainloader, testloader=testloader, maskloader=maskloader, multi_head = multi_head, shuffle_idx=shuffle_idx, use_gpu=use_gpu, log_interval=log_interval, batch_size_train=batch_size_train, connect_ratio=connect_ratio, lr=learning_rate, reg_lambda=reg_lambda)
		elif method == "EWC":
			train_EWC(network, task_num=task_num, n_epochs =n_epochs, trainloader=trainloader, testloader=testloader, maskloader=maskloader, multi_head = multi_head, shuffle_idx=shuffle_idx, use_gpu=use_gpu, log_interval=log_interval, batch_size_train=batch_size_train, connect_ratio=connect_ratio, lr=learning_rate, reg_lambda=reg_lambda, trainset=trainset)
		elif method == "SI":
			print("not implemented yet")

		save_model(network, task_num, multi_head)

		#Inference test on past tasks
		if multi_head:
			for loaded_task in range(task_num + 1):
				network = load_head(network, loaded_task, use_gpu = use_gpu)
				load_mask(network, loaded_task)
				apply_mask(network)
				accuracy= test(network, loaded_task, shuffle_idx, n_epochs, trainloader, testloader, use_gpu)
				save_result(accuracy, loaded_task,save_path, current_task=task_num)
				lift_mask(network)
			network = load_head(network, task_num, use_gpu = use_gpu)
		else:
			for loaded_task in range(task_num + 1):
				load_mask(network, loaded_task)
				apply_mask(network)
				accuracy = test(network, loaded_task, shuffle_idx, n_epochs, trainloader, testloader, use_gpu)
				save_result(accuracy, loaded_task, save_path, current_task=task_num)
				lift_mask(network)


if __name__ =="__main__":
	ISG_train()
	plot_result(num_task, save_path)
	clear_models()
	