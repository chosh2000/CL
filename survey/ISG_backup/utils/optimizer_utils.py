#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import copy
import os
import shutil


class continual_SGD(optim.SGD):
	def __init__(self, params, reg_lambda, lr = 0.001, momentum = 0, dampening = 0, weight_decay = 0, nesterov = False):
		super(continual_SGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
		self.reg_lambda = reg_lambda

	def __setstate__(self, state):
		super(continual_SGD, self).__setstate__(state)

	def step(self, reg_params, closure = None):
		# print("stepping")
		loss = None

		if closure is not None:
			loss = closure()

		
		for group in self.param_groups:
			weight_decay = group['weight_decay']
			momentum = group['momentum']
			dampening = group['dampening']
			nesterov = group['nesterov']

			for i, p in enumerate(group['params']):

				if p.grad is None:
					continue  #ND: proceed to the next for loop iteration. Nothing below here is executed

				d_p = p.grad.data
				# if p in reg_params: #ND: p = parameter id that can also be used as the reg_dictionary key
					# print("param_size: ", p.size(), "i: ", i)

				param_dict = reg_params[i]

				omega = param_dict['omega']
				init_val = param_dict['init_val']
				mask = param_dict['mask']

				curr_param_value = p.data


				# curr_param_value = curr_param_value.cuda() #TODO: uncomment these for CUDA support
				# init_val = init_val.cuda()
				# omega = omega.cuda()

				#get the difference
				param_diff = curr_param_value - init_val

				#get the gradient for the penalty term for change in the weights of the parameters
				local_grad = torch.mul(param_diff, 2*self.reg_lambda*omega)  #lambda*omega = fisher in EWC
				#TODO: should bias have the l_2 penalty?
				
				del param_diff
				del omega
				del init_val
				del curr_param_value
				d_p = (d_p+local_grad)*mask #ND: the mask must be applied to both.
				
				del mask
				del local_grad
					

				if (weight_decay != 0):
					d_p.add_(weight_decay, p.data)

				if (momentum != 0):
					param_state = self.state[p]
					if 'momentum_buffer' not in param_state:
						buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
					else:
						buf = param_state['momentum_buffer']
						buf.mul_(momentum).add_(1 - dampening, d_p)
					if nesterov:
						d_p = d_p.add(momentum, buf)
					else:
						d_p = buf

				p.data.add_(-group['lr'], d_p)

		return loss




class omega_optim(optim.SGD):

	def __init__(self, params, lr = 0.001, momentum = 0, dampening = 0, weight_decay = 0, nesterov = False):
		super(omega_optim, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

	def __setstate__(self, state):
		super(omega_optim, self).__setstate__(state)

	def step(self, reg_params, batch_index, batch_size, use_gpu, closure = None):
		loss = None

		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			weight_decay = group['weight_decay']
			momentum = group['momentum']
			dampening = group['dampening']
			nesterov = group['nesterov']

			for i, p in enumerate(group['params']):

				if p.grad is None:
					continue

				# if p in reg_params:

				# grad_data = p.grad.data

				#The absolute value of the grad_data that is to be added to omega 
				grad_data_copy = p.grad.data.clone()
				grad_data_copy = grad_data_copy.abs()
				
				param_dict = reg_params[i]

				omega = param_dict['omega']
				omega = omega.to(torch.device("cuda:0" if use_gpu else "cpu"))

				prev_size = batch_index * batch_size # batch_index = 0 --> prev_size = 0
				curr_size = (batch_index+1) * batch_size
				omega = omega.mul(prev_size)
				omega = omega.add(grad_data_copy)
				omega = omega.div(curr_size)
				# current_size = (batch_index+1)*batch_size
				# step_size = 1/float(current_size)
				# #Incremental update for the omega
				# omega = omega + step_size*(grad_data_copy - batch_size*(omega))

				param_dict['omega'] = omega

				reg_params[i] = param_dict

		return loss

