import os
import sys
import numpy as np
import torch
from copy import deepcopy
from pathlib import Path


def permute_MNIST(data, shuffle_idx, task_num):
	data = torch.stack([x.view(-1)[shuffle_idx[task_num]].view(1, 28, 28) for x in data], dim=0)
	return data

#pMNIST shuffle
def pMNIST_shuffle(num_task):
	#list of shuffling index for pMNIST
	shuffle_idx = []
	shuffle_idx.append(torch.arange(28*28)) #First take is the original
	for t in range(num_task-1):
		shuffle_idx.append(torch.randperm(28*28))
	return shuffle_idx
