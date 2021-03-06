import torch
from utils.data_prep import *

def SIM_gating(network, task_num, dataloader):
	network.tmodel.eval()
	network.optimizer.zero_grad()

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

	assert network.online_reg, "SIM only supports online reg. computation"

	#Apply masks
	if network.args.random_drop:
		network.random_mask()
	else:
		for n, p in network.tmodel.named_parameters():
			if n in network.tmodel.mask_list:
				a = network.args.alpha
				# b = network.args.beta
				F = p.grad.data.abs() / len(dataloader) #Inductive Fisher
				x = F.max() * network.args.xi
				F = F+x
				if task_num == 0:
					O = torch.zeros(F.shape).to(network.device)
				else:
					O = network.reg_params[0]['importance'][n].clone() #Omega
				# O /= O.max()
				R = F/(O*a+x) #Relevance
				# R = (1-a)*F - a*O #Relevance
				# S = -(1-b)*F - b*O#Secondary, Collateral, Supplement, Auxiliary 
				# S = -1*O#Secondary, Collateral, Supplement, Auxiliary 
				R_sum = R.sum(dim=[i for i in range(1, len(R.shape))])
				# S_sum = S.sum(dim=[i for i in range(1, len(S.shape))])
				R_sum_hist = torch.histc(R_sum, bins=10, min=R_sum.min(), max=R_sum.max())
				M = torch.zeros(network.tmodel.mask_list[n].shape) #Mask initialized to zero


				if network.args.dropmethod == "rho":
					M_index = R_sum.topk(int(R_sum.numel()* network.tmodel.rho[n]))[1]
				# elif network.args.dropmethod == "dist":
				# 	if 'conv1' not in n:
				# 		M_index = R_sum.topk(int(R_sum_hist[network.args.dist_num:10].sum()))[1]
				# 	else:
				# 		M_index = R_sum.topk(R_sum.numel())[1]
				# elif network.args.dropmethod == "random_even":
				# 	if len(network.mask_trace) == 0:
				# 		network.count[n] = torch.zeros(M.shape[0])
				# 	M_index = network.count[n].topk(int(network.count[n].numel() * network.tmodel.rho[n]))[1]
				# 	network.count[n][M_index] -= 1
				else:
					raise "invalid drop method"

				M[M_index] = 1 #Sailent features set to 1
				network.tmodel.mask_list[n].copy_(M) 

				# print("Layer        : {}".format(n))
				# print("R.histogram  : {}".format(R_sum_hist))
				# print("R_sum max/min: {:.1f}/{:.1f}".format(R_sum.max(), R_sum.min()))
				# print("F max  		: {}".format(F.max()))
				# print("O masx 		: {}".format(O.max()))
				# print("F.num_zeros  : {}/{}".format(F.numel()-F.nonzero().size(0), F.numel()))
				# print("W.num_zeros  : {}/{}".format(p.data.numel()-p.data.nonzero().size(0), p.data.numel()))
				# print("O.num_zeros  : {}/{}".format(O.numel()-O.nonzero().size(0), O.numel()))				
	#Saving masks
	network.save_mask(task_num)
	network.optimizer.zero_grad()

