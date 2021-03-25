import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import shutil
import sys
from utils.optimizer_utils import *
from utils.data_prep import *
sys.path.append('./')


class MAS(nn.Module):
	def __init__(self, model, args, ob):
		super(MAS, self).__init__()
		self.args = args
		self.tmodel = model
		self.online_reg = args.online_reg
		self.params = {n: p for n, p in self.tmodel.named_parameters() if p.requires_grad} 
		self.reg_params = {} #{task_num:   {importance:  , task_param: }}  
		self.task_masks = {} #{task_num:   {fc1: 		 , fc2: 	   }}
		self.task_heads = {} #{task_num:   {}}
		self.mask_trace = {}
		self.count      = {}
		#below is for recording results
		self.ob   = ob
		self.ACC  = [] #[[ACC for 1st iteration], [ACC for 2nd iteration],...]
		self.LCA  = []
		self.FWT  = []
		self.BWT  = []
		self.SAT  = []
		self.PTB  = []
		self.IPK  = []
		self.Rii  = {}

		self.criterion_fn = nn.CrossEntropyLoss()
		self.shuffle_idx = pMNIST_shuffle(self.args)
		self.optimizer = optim.Adam(self.tmodel.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.schedule,gamma=0.1)
		self.device = torch.device("cuda:0" if args.use_gpu else "cpu")

	def forward(self, x):
		return self.tmodel.forward(x)

	def criterion(self, data, target):
		output = self.forward(data)
		loss = self.criterion_fn(output, target)
		# loss /= len(target) #TODO: keeps update same scale as reg. term 
		return loss

	def calculate_importance(self, dataloader, task_num):
		assert self.args.method=='MAS', "not using MAS version"
		# Initialize the importance matrix
		if self.online_reg and len(self.reg_params)>0:
			importance = self.reg_params[0]['importance']
		else:
			importance = {}
			for n, p in self.tmodel.named_parameters():
				if 'head' not in n:
					importance[n] = p.clone().detach().fill_(0)  # zero initialized

		mode = self.tmodel.training
		self.tmodel.eval()

		# Accumulate the gradients of L2 loss on the outputs
		for i, (data, target) in enumerate(dataloader):
			if self.args.dataset == "pMNIST":
				data = permute_MNIST(data, self.shuffle_idx, task_num, self.args)
			if self.args.use_gpu:
				data = data.cuda()
				target = target.cuda()
			pred = self.forward(data)
			pred.pow_(2)
			loss = pred.mean()

			self.optimizer.zero_grad()
			loss.backward()
			for n, p in importance.items():
				if self.params[n].grad is not None:
					p += (self.params[n].grad.abs() / len(dataloader)) * self.args.omega_multiplier
		self.tmodel.train(mode=mode)
		return importance

	def update_model(self, data, target):
		assert self.args.method=='MAS', "not using MAS version"
		#Compute loss
		loss = self.criterion(data, target)
		loss = self.add_l2_loss(loss) #applying l2 penalty
		#update
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def finetune_head(self, task_num, trainloader, testloader):
		mode = self.tmodel.training
		self.tmodel.train()

		prev = {}
		for n, p in self.tmodel.named_parameters():
			if 'head' not in n:
				p.requires_grad = False
			else:
				prev[n] = p.clone().detach()

		#train the head layer
		for epoch in range(self.args.finetune_epoch):
			for batch_idx, (data, target) in enumerate(trainloader):
				#Permuting the data
				if self.args.dataset == "pMNIST":
					data = permute_MNIST(data, self.shuffle_idx, task_num, self.args)
				data = data.to(self.device)
				target = target.to(self.device)
				loss = self.criterion(data, target)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
			print("Head Finetuning    :", end='')
			acc = self.test(task_num, testloader, epoch)

		for n, p in self.tmodel.named_parameters():
			p.requires_grad = True
			if 'head' in n:
				if self.args.revert_head:
					with torch.no_grad():
						p.copy_(prev[n])
					
		self.tmodel.train(mode=mode)
		return acc

	def test(self, task_num, testloader, epoch):
		test_losses = []
		test_loss = 0
		correct = 0
		mode = self.tmodel.training
		self.tmodel.eval()
		device = self.device
		self.tmodel.to(device)

		# with torch.no_grad(): # we don't need Autograd for testing. commented out for l2_grad computation.
		for batch_idx, (data, target) in enumerate(testloader):
			#Permute the MNIST data
			if self.args.dataset == "pMNIST":
				data = permute_MNIST(data, self.shuffle_idx, task_num, self.args)
			data = data.to(device)
			target = target.to(device)
			output = self.tmodel.forward(data)

			test_loss += self.criterion_fn(output, target).item()
			pred = output.data.max(1, keepdim=True)[1] #prediction
			correct += pred.eq(target.data.view_as(pred)).sum() #view_as --> use same dim?
		test_loss /= len(testloader.dataset)
		test_losses.append(test_loss)
		accuracy = 100.*correct/len(testloader.dataset)
		if epoch != -1:
			print('Task: {},\t Epoch: {}/{},\t Avg.loss: {:.4f},\t Test Accuracy: {}/{}({:.0f}%)'.format(
			task_num, epoch, self.args.schedule[-1], test_loss, correct, len(testloader.dataset), accuracy))

		self.tmodel.train(mode=mode)
		return accuracy

	def task_parameter(self):
		task_param = {}
		for n, p in self.tmodel.named_parameters():
			if "head" not in n:
				task_param[n] = p.clone().detach()
		return task_param

	def add_l2_loss(self, loss):
		if len(self.reg_params)>0:
			if self.online_reg == False:
				reg_loss = 0
				for i, reg_term in self.reg_params.items():
					task_reg_loss = 0
					importance = reg_term['importance']
					task_param = reg_term['task_param']
					for n, p in self.tmodel.named_parameters():
						if "head" not in n:
							task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
					reg_loss += task_reg_loss
				loss += self.args.reglambda * reg_loss
			else:  #online regularization
				reg_loss = 0
				importance = self.reg_params[0]['importance'] # this is the most recent omega (tasknum - 1)
				task_param = self.reg_params[0]['task_param']
				for n, p in self.tmodel.named_parameters():
					if "head" not in n:
						reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
				loss += self.args.reglambda * reg_loss
		return loss

	#Head functions
	def save_head(self, task_num):
		assert self.args.multi_head,  "must be a multi_head approach"
		head_weight = self.tmodel.fc_head.weight.data.clone().detach()
		head_bias = self.tmodel.fc_head.bias.data.clone().detach()
		self.task_heads[task_num] = {'weight': head_weight, 'bias':head_bias}

	def load_head(self, task_num): #only used during the inference step of multi-head method. Single head doesn't need save/load.
		assert self.args.multi_head,  "must be a multi_head approach"
		with torch.no_grad():
			self.tmodel.fc_head.weight.copy_(self.task_heads[task_num]['weight'])
			self.tmodel.fc_head.bias.copy_(self.task_heads[task_num]['bias'])

	def load_new_head(self):
		new = nn.Linear(self.tmodel.fc_head.in_features, self.tmodel.fc_head.out_features).to(self.device)
		self.tmodel.fc_head.load_state_dict(new.state_dict())


	#Mask functions
	def save_mask(self, task_num):
		self.task_masks[task_num] = {}
		for n,p in self.tmodel.mask_list.items():
			self.task_masks[task_num][n] = self.tmodel.mask_list[n].clone().detach()


	def load_mask(self, task_num):
		for n,p in self.tmodel.mask_list.items():
			self.tmodel.mask_list[n].copy_(self.task_masks[task_num][n])


	def lift_mask(self):
		for n,p in self.tmodel.mask_list.items():
			self.tmodel.mask_list[n].fill_(1)

	def random_mask(self):
		for n,p in self.tmodel.mask_list.items():
			mask = torch.FloatTensor(p.shape).uniform_() > (1-self.tmodel.rho[n])
			p.copy_(mask)	


	def record_trace(self):
		# Keep a record of all neurons used in past.
		if len(self.mask_trace) == 0:
			for n,p in self.tmodel.mask_list.items():
				self.mask_trace[n] = p.clone()
		else:
			for n,p in self.tmodel.mask_list.items():
				self.mask_trace[n] += p
		#print
		print("Mask histogram")
		for n,p in self.mask_trace.items():
			print("{}: \t {}".format(n, torch.histc(p, bins=11, min=0, max=11)))




class SI(MAS):
	"""
	@inproceedings{zenke2017continual,
		title={Continual Learning Through Synaptic Intelligence},
		author={Zenke, Friedemann and Poole, Ben and Ganguli, Surya},
		booktitle={International Conference on Machine Learning},
		year={2017},
		url={https://arxiv.org/abs/1703.04200}
	}
	"""
	def __init__(self, model, args, ob):
		super(SI, self).__init__(model, args, ob)
		self.damping_factor = 0.1
		self.w = {}
		self.initial_params = {}

		for n, p in self.params.items():
			self.w[n] = p.clone().detach().zero_().cuda()
			self.initial_params[n] = p.clone().detach().cuda() # The initial_params will only be used in the first task


	def update_model(self, data, target):
		assert self.args.method=='SI', "not using SI version" 
		unreg_gradients = {}
		
		# 1.Save current parameters
		old_params = {}
		for n, p in self.params.items():
			old_params[n] = p.clone().detach()

		# 2. Collect the gradients without regularization term
		loss = self.criterion(data, target)
		self.optimizer.zero_grad()
		loss.backward(retain_graph=True)
		for n, p in self.params.items():
			if p.grad is not None:
				unreg_gradients[n] = p.grad.clone().detach()

		# 3. Normal update with regularization
		loss = self.criterion(data, target)
		loss = self.add_l2_loss(loss) #applying l2 penalty
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# 4. Accumulate the w
		for n, p in self.params.items():
			delta = p.detach() - old_params[n]
			if n in unreg_gradients.keys():  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
				self.w[n] -= unreg_gradients[n] * delta  # w[n] is >=0

		# return loss.detach(), out

	def calculate_importance(self, dataloader, task_num):
		assert self.args.method=='SI', "not using SI version" 
		# Initialize the importance matrix
		if self.online_reg and len(self.reg_params)>0:
			importance  = self.reg_params[0]['importance']
			prev_params = self.reg_params[0]['task_param']
		else:
			importance = {}
			for n, p in self.tmodel.named_parameters():
				if 'head' not in n:
					importance[n] = p.clone().detach().fill_(0)  # zero initialized
			prev_params = self.initial_params

		# mode =self.tmodel.training
		# self.tmodel.eval()

		# # Data prep.
		# for i, (data, target) in enumerate(dataloader):
		# 	if self.args.dataset == "pMNIST":
		# 		data   = permute_MNIST(data, self.shuffle_idx, task_num, self.args)
		# 	if self.args.use_gpu:
		# 		data   = data.cuda()
		# 		target = target.cuda()

		# Calculate or accumulate the Omega (the importance matrix)
		for n, p in importance.items():
			delta_theta = self.params[n].detach() - prev_params[n]
			p += self.w[n]/(delta_theta**2 + self.damping_factor)
			self.w[n].zero_()
		# self.tmodel.train(mode = mode)

		return importance





class EWC(MAS):
	"""
	@article{kirkpatrick2017overcoming,
		title={Overcoming catastrophic forgetting in neural networks},
		author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
		journal={Proceedings of the national academy of sciences},
		year={2017},
		url={https://arxiv.org/abs/1612.00796}
	}
	"""

	def __init__(self, model, args, ob):
		super(EWC, self).__init__(model, args, ob)
		# self.online_reg = False
		self.n_fisher_sample = None
		self.empFI = False


	def update_model(self, data, target):
		assert self.args.method=='EWC', "not using EWC version"
		#Compute loss
		loss = self.criterion(data, target)
		loss = self.add_l2_loss(loss) #applying l2 penalty
		#update
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def calculate_importance(self, dataloader, tasknum):
		assert self.args.method=='EWC', "not using EWC version" 

		# Update the diag fisher information
		# There are several ways to estimate the F matrix.
		# We keep the implementation as simple as possible while maintaining a similar performance to the literature.
		# self.log('Computing EWC')

		# Initialize the importance matrix
		if self.online_reg and len(self.reg_params)>0:
			importance = self.reg_params[0]['importance']
		else:
			importance = {}
			for n, p in self.params.items():
				importance[n] = p.clone().detach().fill_(0)  # zero initialized

		# Sample a subset (n_fisher_sample) of data to estimate the fisher information (batch_size=1)
		# Otherwise it uses mini-batches for the estimation. This speeds up the process a lot with similar performance.
		if self.n_fisher_sample is not None:
			n_sample = min(self.n_fisher_sample, len(dataloader.dataset))
			self.log('Sample',self.n_fisher_sample,'for estimating the F matrix.')
			rand_ind = random.sample(list(range(len(dataloader.dataset))), n_sample)
			subdata = torch.utils.data.Subset(dataloader.dataset, rand_ind)
			dataloader = torch.utils.data.DataLoader(subdata, shuffle=True, num_workers=2, batch_size=1)

		mode = self.tmodel.training
		self.tmodel.eval()

		# Accumulate the square of gradients
		for i, (data, target) in enumerate(dataloader):
			if self.args.dataset == "pMNIST":
				data   = permute_MNIST(data, self.shuffle_idx, task_num, self.args)
			if self.args.use_gpu:
				data   = data.cuda()
				target = target.cuda()


			preds = self.forward(data)

			# # Sample the labels for estimating the gradients
			# # For multi-headed model, the batch of data will be from the same task,
			# # so we just use task[0] as the task name to fetch corresponding predictions
			# # For single-headed model, just use the max of predictions from preds['All']
			# task_name = task[0] if self.multihead else 'All'

			# # The flag self.valid_out_dim is for handling the case of incremental class learning.
			# # if self.valid_out_dim is an integer, it means only the first 'self.valid_out_dim' dimensions are used
			# # in calculating the loss.
			# pred = preds[task_name] if not isinstance(self.valid_out_dim, int) else preds[task_name][:,:self.valid_out_dim]

			ind = preds.max(1)[1].flatten()  # Choose the one with max
			if self.args.use_gpu:
				ind = ind.cuda()

			# # - Alternative ind by multinomial sampling. Its performance is similar. -
			# # prob = torch.nn.functional.softmax(preds['All'],dim=1)
			# # ind = torch.multinomial(prob,1).flatten()

			if self.empFI:  # Use groundtruth label (default is without this)
				ind = target

			loss = self.criterion(data, ind)
			self.optimizer.zero_grad()
			loss.backward()
			for n, p in importance.items():
				if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
					p += ((self.params[n].grad ** 2) * len(data) / len(dataloader))

		self.tmodel.train(mode = mode)


		return importance



def clear_models():
	model_path = os.path.join(os.getcwd(), "models")
	shutil.rmtree(model_path)
