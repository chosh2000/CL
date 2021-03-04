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
	n_epochs = args.schedule[-1]
	for epoch in range(n_epochs+1):
		network.scheduler.step(epoch) #TODO: This might cause poor training in later tasks
		for batch_idx, (data, target) in enumerate(trainloader):
			data = data.to(network.device)
			target = target.to(network.device)

			#Compute loss
			loss = network.criterion(data, target)

			#update
			network.optimizer.zero_grad()
			loss.backward()
			network.optimizer.step()

		network.test(task_num, testloader, epoch)

def train(network, args, task_num, trainloader, testloader, maskloader=None):
	train_losses = []
	train_counter = []
	network.tmodel.to(network.device)


	#Compute Inductive Fisher and apply SIM mask
	if args.apply_SIM:
		SIM_gating(network, task_num, trainloader)
	network.optimizer.zero_grad()
	network.record_trace()

	#Training Phase
	network.load_new_head()
	print(network.tmodel.training)
	p()
	network.tmodel.train()

	#Finetuning
	acc_init = network.finetune_head(task_num, trainloader, testloader)

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

			#6. make measurementes
			omega_sum = 0
			for n, p in network.tmodel.named_parameters():
				if 'head' not in n:
					omega_sum += importance[n].sum()
			network.SAT.append(omega_sum)
			network.FWT.append(acc_int/acc)
			network.Rii[task_num] = acc

		else:
			#1. Learn the parameters for the current task
			for batch_idx, (data, target) in enumerate(trainloader):
				#Permuting the data
				if network.args.dataset == "pMNIST":
					data = permute_MNIST(data, network.shuffle_idx, task_num, args)

				data = data.to(network.device)
				target = target.to(network.device)

				network.update_model(data, target)

			#Test at the end of each epoch
			acc = network.test(task_num, testloader, epoch)


