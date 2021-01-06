import torch
from utils.data_prep import *

def SIM_gating(network, task_num, dataloader):
	network.tmodel.eval()
	network.optimizer.zero_grad()
	# network.lift_mask()
	# network.lift_mask() #Make sure no mask is applied yet
	#Populate Inductive Fisher
	for batch_idx, (data, target) in enumerate(dataloader):
		if network.args.dataset == "pMNIST":
			data = permute_MNIST(data, network.shuffle_idx, task_num, network.args)
		data = data.to(network.device)
		target = target.to(network.device)

		#get the function outputs
		outputs = network.tmodel.SIM_forward(data)
		outputs.pow_(2)
		loss = outputs.mean()
		#compute gradients for these parameters
		loss.backward()

	#Apply masks
	assert network.online_reg, "SIM only supports online reg. computation"
	# if task_num == 0:
	# 	network.random_mask()
	if task_num >= 0  :
		network.stat[task_num] = {'omega_sum': {}}
		for n, p in network.tmodel.named_parameters():
			# if "head" not in n and "bias" not in n:
			if n in network.tmodel.mask_list:
				a = network.args.alpha
				x = network.args.xi 
				F = p.grad.data.abs() / len(dataloader) #Inductive Fisher
				F /= F.max()
				if task_num == 0:
					O = torch.ones(F.shape).to(network.device)
				else:
					O = network.reg_params[0]['importance'][n].clone() #Omega
				O /= O.max()
				# R = F/(O*a+x) #Relevance
				R = F-O*a #Relevance
				network.stat[task_num]['omega_sum'][n] = O.sum()
				# print("Xi           : {}".format(x))
				print("Layer        : {}".format(n))
				print("Sum Omega    : {}".format(O.sum()))
				print("W.num_zeros  : {}/{}".format(p.data.numel()-p.data.nonzero().size(0), p.data.numel()))
				print("F.num_zeros  : {}/{}".format(F.numel()-F.nonzero().size(0), F.numel()))
				print("O.num_zeros  : {}/{}".format(O.numel()-O.nonzero().size(0), O.numel()))				
				print("Max Fisher   : {}".format(F.max()))
				print("Max Omega    : {}".format(O.max()))
				print("Min Fisher   : {}".format(F.min()))
				print("Min Omega    : {}".format(O.min()))
				print("Max Relevance: {}".format(R.max()))
				M = torch.zeros(network.tmodel.mask_list[n].shape) #Mask initialized to zero
				R_sum = R.sum(dim=[i for i in range(1, len(R.shape))])
				M_index = R_sum.topk(int(R_sum.numel()* network.tmodel.rho[n]))[1]
				M[M_index] = 1 #Sailent features set to 1
				# print("Mask active  : {}/{}\n".format(M.sum(), M.numel()))
				assert network.tmodel.mask_list[n] is not None, "Mask mismatch" #
				if network.tmodel.rho[n] < 1:
					assert M.sum() < M.numel(), "Mask error"
				# network.tmodel.mask_list[n].data = M.to(network.device) ##TODO: fix
				network.tmodel.mask_list[n].copy_(M) ##TODO: fix here

	#Saving masks
	network.save_mask(task_num)
	network.optimizer.zero_grad()

