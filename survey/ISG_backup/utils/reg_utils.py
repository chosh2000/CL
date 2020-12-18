#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import copy
import os
import shutil
import sys

from utils.model_utils import *



def init_reg_params(model, use_gpu, freeze_layers = []):
	"""
	Input:
	Output:
	Function: 	
	"""
	device = torch.device("cuda:0" if use_gpu else "cpu")
	reg_params = {}

	for i, (name, param) in enumerate(model.tmodel.named_parameters()):
		# if not name in freeze_layers:
		param_dict = {}
		print ("Initializing omega values for layer {}     size: {}".format(name,param.size()))
		omega = torch.zeros(param.size()).to(device)
		pomega = torch.zeros(param.size()).to(device)
		mask = torch.ones(param.size()).to(device) # ND: ones rather than zeros for the bias.
		init_val = param.data.clone()

		#for first task, omega is initialized to zero
		param_dict['omega'] = omega
		param_dict['prev_omega'] = pomega
		param_dict['init_val'] = init_val
		param_dict['mask'] = mask
		#the key for this dictionary is the name of the layer
		del omega
		del pomega
		del init_val
		del mask

		reg_params[i] = param_dict

	model.reg_params = reg_params

	return model


def init_reg_params_across_tasks(model, use_gpu, multi_head, freeze_layers = []):
	"""
	Input:
	1) model: A reference to the model that is being trained
	2) use_gpu: Set the flag to True if the model is to be trained on the GPU
	Output:
	Function:
	"""

	#Get the reg_params for the model 
	
	device = torch.device("cuda:0" if use_gpu else "cpu")
	reg_params = model.reg_params

	for i, (name, param) in enumerate(model.tmodel.named_parameters()):
		param_dict = reg_params[i]
		print ("Initializing omega values for layer {}     size: {}".format(name,param.size()))

		if "fc_head" in name and multi_head:
			# Initialize a new omega
			new_omega = torch.zeros(param.size())
			new_omega = new_omega.to(device)
			mask = torch.ones(param.size())
			mask = mask.to(device)
			# param_dict['prev_omega'] = prev_omega
			param_dict['prev_omega'] = param_dict['omega'].clone().to(device)
			param_dict['omega'] = new_omega
			param_dict['mask'] = mask
			del new_omega
			del mask

		else:
			#Store the previous values of omega			
			mask = torch.ones(param.size())
			mask = mask.to(device)
			init_val = param.data.clone()
			init_val = init_val.to(device)
			omega = param_dict['omega'].to(device)

			#store the initial values of the parameters
			param_dict['omega'] = omega
			param_dict['prev_omega'] = omega.clone()
			param_dict['init_val'] = init_val
			param_dict['mask'] = mask
			del omega
			del init_val
			del mask
		#the key for this dictionary is the name of the layer
		reg_params[i] =  param_dict

	model.reg_params = reg_params
	return model

def accumulate_omega(model, use_gpu):
	device = torch.device("cuda:0" if use_gpu else "cpu")
	reg_params = model.reg_params

	for i, (name, param) in enumerate(model.tmodel.named_parameters()):
		param_dict = reg_params[i]
		print("Accumulating the omega for layer {}".format(name))
		prev_omega = param_dict['prev_omega'].to(device)
		new_omega = param_dict['omega'].to(device)
		acc_omega = torch.add(prev_omega, new_omega)
		#update
		param_dict['omega'] = acc_omega
		del prev_omega
		del new_omega
		del acc_omega
		reg_params[i] = param_dict
		
	model.reg_params = reg_params
	return model



def compute_l2_grads_norm(model, inputs, optimizer, target, use_gpu):
	"""
	Inputs:
	Outputs:
	Function:
	"""
	device = torch.device("cuda:0" if use_gpu else "cpu")
	# inputs = inputs.to(device)

	#Zero the parameter gradients
	optimizer.zero_grad()

	#get the function outputs
	outputs = model.tmodel(inputs)
	del inputs

	# BP the error as well
	# loss = model_criterion(outputs, target) #TODO: used for the combined loss

	#compute the sqaured l2 norm of the function outputs
	l2_norm = torch.norm(outputs, 2, dim = 1)
	del outputs

	squared_l2_norm = l2_norm**2
	del l2_norm
	
	sum_norm = torch.sum(squared_l2_norm)
	# sum_norm = torch.sum(squared_l2_norm) + loss  #TODO: used for the combined loss
	del squared_l2_norm



	#compute gradients for these parameters
	sum_norm.backward()

	return model


#Intelligent gating implementation
def intelligent_gating(model, task_num, use_gpu, multi_head, connect_ratio):
	device = torch.device("cuda:0" if use_gpu else "cpu")
	for i, (name, params) in enumerate(model.tmodel.named_parameters()):
		# model.orig_params.append(params.clone()) #saves original parameters before dropout
		with torch.no_grad(): # this is necessary for copy_() below to work
			# if "bias" not in name:
				if "head" not in name:
					model.orig_params[name] = params.clone()
					drop_mask = torch.zeros(params.grad.numel())
					drop_mask[params.grad.abs().flatten().topk(int(params.grad.numel()* connect_ratio))[1]] = 1 #select top k gradients
					drop_mask = drop_mask.reshape(params.grad.size()).to(device)
					model.reg_params[i]['mask'].copy_(drop_mask)
					params.copy_(params * drop_mask) #drops weak connections
					del drop_mask
				if "head" in name and not multi_head:
					model.orig_params[name] = params.clone()
					drop_mask = torch.zeros(params.grad.numel())
					drop_mask[params.grad.abs().flatten().topk(int(params.grad.numel()* connect_ratio))[1]] = 1 #select top k gradients
					drop_mask = drop_mask.reshape(params.grad.size()).to(device)
					model.reg_params[i]['mask'].copy_(drop_mask)
					params.copy_(params * drop_mask) #drops weak connections
					del drop_mask
	return model


def preserve_dropped_params(model):
	reg_params = model.reg_params
	for i, (name, params) in enumerate(model.tmodel.named_parameters()):
		if name in model.orig_params: #if this is a dropped parameter
			param_dict = reg_params[i]
			mask = param_dict['mask']
			orig_params_to_restore = model.orig_params[name] * (1 - mask)
			with torch.no_grad():   #TODO: make sure this does not affect the gradient computation
				params.copy_(orig_params_to_restore + params * mask)
	model.orig_params = {} #clear this out for memory
	return model

def apply_mask(model):
	for i, (name, params) in enumerate(model.tmodel.named_parameters()):
		model.orig_params[name] = params.clone() #save the original parameters
		mask = model.reg_params[i]['mask']
		# params.copy_(params * mask)
		with torch.no_grad():
			params.copy_(params * mask) #apply the mask to params

def lift_mask(model):
	for i, (name, params) in enumerate(model.tmodel.named_parameters()):
		with torch.no_grad():
			params.copy_(model.orig_params[name])
	model.orig_params = {}



def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=20):
	"""
	Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
	
	"""
	lr = init_lr * (0.1**(epoch // lr_decay_epoch))
	print('lr is '+str(lr))

	if (epoch % lr_decay_epoch == 0):
		print('LR is set to {}'.format(lr))

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	return optimizer
