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
	def __init__(self, model, args):
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
    def __init__(self, model, args):
        super(SI, self).__init__(model, args)
        self.damping_factor = 0.1
        self.w = {}
        self.initial_params = {}

        for n, p in self.params.items():
            self.w[n] = p.clone().detach().zero_()
            self.initial_params[n] = p.clone().detach() # The initial_params will only be used in the first task


    def update_model(self, inputs, targets, tasks):

        unreg_gradients = {}
        
        # 1.Save current parameters
        old_params = {}
        for n, p in self.params.items():
            old_params[n] = p.clone().detach()

        # 2. Collect the gradients without regularization term
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks, regularization=False)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        for n, p in self.params.items():
            if p.grad is not None:
                unreg_gradients[n] = p.grad.clone().detach()

        # 3. Normal update with regularization
        loss = self.criterion(out, targets, tasks, regularization=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. Accumulate the w
        for n, p in self.params.items():
            delta = p.detach() - old_params[n]
            if n in unreg_gradients.keys():  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
                self.w[n] -= unreg_gradients[n] * delta  # w[n] is >=0

        return loss.detach(), out

    def calculate_importance(self, dataloader):
        self.log('Computing SI')
        assert self.online_reg,'SI needs online_reg=True'

        # Initialize the importance matrix
        if len(self.regularization_terms)>0: # The case of after the first task
            importance = self.regularization_terms[1]['importance']
            prev_params = self.regularization_terms[1]['task_param']
        else:  # It is in the first task
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)  # zero initialized
            prev_params = self.initial_params

        # Calculate or accumulate the Omega (the importance matrix)
        for n, p in importance.items():
            delta_theta = self.params[n].detach() - prev_params[n]
            p += self.w[n]/(delta_theta**2 + self.damping_factor)
            self.w[n].zero_()

        return importance






def clear_models():
	model_path = os.path.join(os.getcwd(), "models")
	shutil.rmtree(model_path)
