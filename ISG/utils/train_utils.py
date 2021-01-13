import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.optimizer_utils import *
from utils.reg_utils import *
from utils.model_utils import *
from utils.data_prep import *
from utils.EWC_utils import *

def init_train(network, args, task_num, trainloader, testloader, maskloader=None):
	train_losses = []
	train_counter = []
	network.tmodel.to(network.device)
	network.tmodel.train()
	n_epochs = 40
	for epoch in range(n_epochs+1):
		for batch_idx, (data, target) in enumerate(trainloader):
			data = data.to(network.device)
			target = target.to(network.device)

			#Compute loss
			loss = network.criterion(data, target)

			#update
			network.optimizer.zero_grad()
			loss.backward()
			network.optimizer.step()

		test(network, task_num, testloader, epoch)

def train_SIM(network, args, task_num, trainloader, testloader, maskloader=None):
	train_losses = []
	train_counter = []
	#Set network to train
	network.tmodel.to(network.device)


	#Compute Inductive Fisher and apply SIM mask
	SIM_gating(network, task_num, trainloader)
	network.optimizer.zero_grad()

	#Training Phase
	network.load_new_head()
	network.record_trace()
	# print(network.tmodel.fc_head.weight.sum())

	network.tmodel.train()
	n_epochs = args.schedule[-1]
	for epoch in range(n_epochs+1):
		# network.log('Epoch:{0}'.format(epoch))
		network.scheduler.step(epoch) #TODO: This might cause poor training in later tasks

		#updating omega
		if epoch == n_epochs:
			#2. Backup the weight of current task
			network.save_head(task_num)
			task_param = network.task_parameter()
			#3. calculate the importance for this task
			importance = network.calculate_importance(trainloader, task_num)
			#4. copy them to reg_params
			if network.online_reg and len(network.reg_params) > 0:
				# only one slot is used to record the reg data
				network.reg_params[0] = {'importance': importance, 'task_param': task_param}
			else:
				# task-specific parameters are all recorded
				network.reg_params[task_num] = {'importance': importance, 'task_param': task_param}

			#5. restore drop
			network.lift_mask()

		else:
			#1. Learn the parameters for the current task
			for batch_idx, (data, target) in enumerate(trainloader):
				#Permuting the data
				if network.args.dataset == "pMNIST":
					data = permute_MNIST(data, network.shuffle_idx, task_num, args)

				data = data.to(network.device)
				target = target.to(network.device)

				#Compute loss
				loss = network.criterion(data, target)
				loss = network.add_l2_loss(loss) #applying l2 penalty

				#update
				network.optimizer.zero_grad()
				loss.backward()
				network.optimizer.step()

			#Test at the end of each epoch
			test(network, task_num, testloader, epoch)

def test(network, task_num, testloader, epoch):
	test_losses = []
	test_loss = 0
	correct = 0
	# network.eval()
	device = network.device
	network.tmodel.to(device)

	# with torch.no_grad(): # we don't need Autograd for testing. commented out for l2_grad computation.
	for batch_idx, (data, target) in enumerate(testloader):
		#Permute the MNIST data
		if network.args.dataset == "pMNIST":
			data = permute_MNIST(data, network.shuffle_idx, task_num, network.args)
		data = data.to(device)
		target = target.to(device)
		output = network.tmodel.forward(data)

		test_loss += network.criterion_fn(output, target).item()
		pred = output.data.max(1, keepdim=True)[1] #prediction
		correct += pred.eq(target.data.view_as(pred)).sum() #view_as --> use same dim?
	test_loss /= len(testloader.dataset)
	test_losses.append(test_loss)
	accuracy = 100.*correct/len(testloader.dataset)
	print('Task: {},\t Epoch: {}/{},\t Avg.loss: {:.4f},\t Accuracy: {}/{}({:.0f}%)'.format(
		task_num, epoch, network.args.schedule[-1], test_loss, correct, len(testloader.dataset), accuracy))
	return accuracy

