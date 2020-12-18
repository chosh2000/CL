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


def train_MAS(network, task_num, n_epochs, trainloader, testloader, maskloader, multi_head ,shuffle_idx, use_gpu, log_interval, batch_size_train, connect_ratio, lr = 0.001, reg_lambda=0.01):
	train_losses = []
	train_counter = []
	#Set network to train
	device = torch.device("cuda:0" if use_gpu else "cpu")
	network.tmodel.train()
	network.tmodel.to(device)
	optimizer = continual_SGD(network.tmodel.parameters(), reg_lambda, lr = lr) #added for MAS

	#Compute the task-associated ISG mask
	for batch_idx, (data, target) in enumerate(maskloader):
		data = permute_MNIST(data, shuffle_idx, task_num)
		data = data.to(device)
		target = target.to(device)
		network = compute_l2_grads_norm(network, data, optimizer, target, use_gpu)
		network = intelligent_gating(network, task_num, use_gpu, multi_head, connect_ratio)
		save_mask(network, task_num)
		optimizer.zero_grad()

	for epoch in range(n_epochs+1):
		#updating omega
		if epoch == n_epochs:
			omega_optimizer = omega_optim(network.tmodel.parameters(), lr=0.0001) #momentum=0.9
			for batch_idx, (data, target) in enumerate(testloader):
				data = permute_MNIST(data, shuffle_idx, task_num)
				data = data.to(device)
				target = target.to(device)
				network = compute_l2_grads_norm(network, data, omega_optimizer, target, use_gpu) #optimizer populates gardients at this point
				omega_optimizer.step(network.reg_params, batch_idx, target.size(0), use_gpu) #First sets omega=0, then does moving avg. for this task.
			network = accumulate_omega(network, use_gpu)
			network = preserve_dropped_params(network)

		else:
			# optimizer = exp_lr_scheduler(optimizer, epoch, init_lr=lr, lr_decay_epoch=20) # exponential lr decay
			for batch_idx, (data, target) in enumerate(trainloader):
				#Permuting the data
				data = permute_MNIST(data, shuffle_idx, task_num)
				data = data.to(device)
				target = target.to(device)

				optimizer.zero_grad()
				output = network.forward(data)
				loss = model_criterion(output, target)
				loss.backward()
				optimizer.step(network.reg_params) #using continual_SGD optimizer

				#Train stats
				if batch_idx % log_interval == 0:
					print("Task: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(task_num,
						epoch, batch_idx * len(data), len(trainloader.dataset), 100. * batch_idx / len(trainloader), loss.item() ))
					train_losses.append(loss.item())
					train_counter.append(
						(batch_idx*batch_size_train)+((epoch-1)*len(trainloader.dataset)))

			#Test at the end of each epoch
			test(network, task_num=task_num, shuffle_idx=shuffle_idx, n_epochs=n_epochs, trainloader=trainloader, testloader=testloader, use_gpu=use_gpu)


def train_EWC(network, task_num, n_epochs, trainloader, testloader, maskloader, multi_head ,shuffle_idx, use_gpu, log_interval, batch_size_train, trainset, connect_ratio, lr = 0.001, reg_lambda=0.01):
	train_losses = []
	train_counter = []
	#Set network to train
	device = torch.device("cuda:0" if use_gpu else "cpu")
	network.tmodel.train()
	network.tmodel.to(device)
	optimizer = continual_SGD(network.tmodel.parameters(), reg_lambda, lr = lr) #added for MAS

	#Compute the task-associated ISG mask
	for batch_idx, (data, target) in enumerate(maskloader):
		data = permute_MNIST(data, shuffle_idx, task_num)
		data = data.to(device)
		target = target.to(device)
		network = compute_l2_grads_norm(network, data, optimizer, target, use_gpu)
		network = intelligent_gating(network, task_num, use_gpu, multi_head, connect_ratio)
		save_mask(network, task_num)
		optimizer.zero_grad()


	for epoch in range(n_epochs+1):

		#updating omega
		if epoch == n_epochs:
			network = EWC_update_fisher_params(network, trainset, use_gpu, shuffle_idx, task_num, batch_size=100, num_batch=300)
			network = accumulate_omega(network, use_gpu)
			network = preserve_dropped_params(network)
		else:
			# optimizer = exp_lr_scheduler(optimizer, epoch, init_lr=lr, lr_decay_epoch=20) # exponential lr decay
			for batch_idx, (data, target) in enumerate(trainloader):
				#Permuting the data
				data = permute_MNIST(data, shuffle_idx, task_num)
				data = data.to(device)
				target = target.to(device)

				#standard training using the dropped network
				optimizer.zero_grad()
				output = network.forward(data)
				loss = model_criterion(output, target)
				loss.backward()
				optimizer.step(network.reg_params) #continual_SGD optimizer

				#Train stats
				if batch_idx % log_interval == 0:
					print("Task: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(task_num,
						epoch, batch_idx * len(data), len(trainloader.dataset), 100. * batch_idx / len(trainloader), loss.item() ))
					train_losses.append(loss.item())
					train_counter.append(
						(batch_idx*batch_size_train)+((epoch-1)*len(trainloader.dataset)))

			#Test at the end of each epoch
			test(network, task_num=task_num, shuffle_idx=shuffle_idx, n_epochs=n_epochs, trainloader=trainloader, testloader=testloader, use_gpu=use_gpu)


def test(network, task_num, shuffle_idx, n_epochs, trainloader, testloader, use_gpu):
	test_losses = []
	# test_counter = [i*len(trainloader.dataset) for i in range(n_epochs + 1)]
	test_loss = 0
	correct = 0
	# network.eval()
	# network.tmodel.train()  # required for l2_grad computation via BP
	device = torch.device("cuda:0" if use_gpu else "cpu")
	network.tmodel.to(device)
	# network.tmodel = network.tmodel.cuda()
	# optimizer = optim.SGD(network.tmodel.parameters(), lr = 0.001)
	# optimizer = continual_SGD(network.tmodel.parameters(), reg_lambda=0.01, lr = 0.001) #added for MAS
	
	# for i, (name, params) in enumerate(network.tmodel.named_parameters()):
	# 	print(params.device)
	# p()
	# with torch.no_grad(): # we don't need Autograd for testing. commented out for l2_grad computation.
	for batch_idx, (data, target) in enumerate(testloader):
		#Permute the MNIST data
		data = permute_MNIST(data, shuffle_idx, task_num)
		data = data.to(device)
		target = target.to(device)
		output = network.tmodel.forward(data)


		test_loss += model_criterion(output, target).item()
		pred = output.data.max(1, keepdim=True)[1] #prediction
		correct += pred.eq(target.data.view_as(pred)).sum() #view_as --> use same dim?
	test_loss /= len(testloader.dataset)
	test_losses.append(test_loss)
	accuracy = 100.*correct/len(testloader.dataset)
	print('\nTask: {}   Test set: Avg. loss: {:.4f}, Accuracy: {}/{}({:.0f}%)\n'.format(
		task_num, test_loss, correct, len(testloader.dataset), accuracy))
	return accuracy
