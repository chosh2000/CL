from __future__ import print_function

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import os
import shutil
import pickle
import sys
import glob
import shutil
sys.path.append('./')


class shared_model(nn.Module):

	def __init__(self, model):
		super(shared_model, self).__init__()
		# self.tmodel = models.alexnet(pretrained = True)
		self.tmodel = model
		self.reg_params = {}
		self.orig_params = {}
	def forward(self, x):
		return self.tmodel(x)


#Last linear layer of the model as buffer
class classification_head(nn.Module):
	def __init__(self, in_features, out_features):
		super(classification_head, self).__init__()
		self.fc = nn.Linear(in_features, out_features)
	def forward(self, x):
		return x
		

def model_criterion(preds, labels):
	loss =  nn.CrossEntropyLoss()
	return loss(preds, labels)

def model_init(model, task_num, num_classes, multi_head, use_gpu = False):
	#model path
	path = os.path.join(os.getcwd(), "models")
	if not os.path.exists(path):
		os.mkdir(path)
	#Different model/reg paths
	if multi_head:
		#multihead model path
		model_path = os.path.join(path, "multihead_shared_model.pth")
		reg_path = os.path.join(path, "multihead_shared_reg_params.pickle")
	else:
		#singlehead model path
		model_path = os.path.join(path, "singlehead_shared_model.pth")
		reg_path = os.path.join(path, "singlehead_shared_reg_params.pickle")

	#load model
	if os.path.isfile(model_path) and task_num > 0:
		print("Loading most recent(task:{}) model parameters: {}".format(task_num-1,model_path))
		model.load_state_dict(torch.load(model_path))
	if os.path.isfile(reg_path) and task_num > 0:
		print("Loading most recent(task:{}) reg. parameters: {}".format(task_num-1,reg_path))
		with open(reg_path, 'rb') as h:
			reg_params = pickle.load(h)
		model.reg_params = reg_params
	#replace to new head
	##Question: should we load an entirely new head or use previous head? previous head.
	if multi_head:
		in_feat = model.tmodel.fc_head.in_features
		model.tmodel.fc_head = nn.Linear(in_feat, num_classes)

	device = torch.device("cuda:0" if use_gpu else "cpu")
	model.train(True)
	model.to(device)
	return model

def save_model(model, task_num, multi_head):
	#model path
	model_path = os.path.join(os.getcwd(), "models")

	#Saving head buffer, only for multi_head method 
	if multi_head:
		head_path = os.path.join(model_path, "task_" + str(task_num)) #only for multi-head
		if not os.path.exists(head_path):
			os.mkdir(head_path)
		head_buffer = classification_head(model.tmodel.fc_head.in_features, model.tmodel.fc_head.out_features)
		head_buffer.fc.weight.data = model.tmodel.fc_head.weight.data
		head_buffer.fc.bias.data = model.tmodel.fc_head.bias.data
		torch.save(head_buffer.state_dict(), os.path.join(head_path, "head.pth"))
		del head_buffer
		#saving regularization parameters
		reg_params = model.reg_params
		f = open(os.path.join(model_path, "multihead_shared_reg_params.pickle"), 'wb')
		pickle.dump(reg_params, f)
		f.close()
		# g = open(os.path.join(head_path, "multihead_reg_params.pickle"), 'wb')
		# pickle.dump(reg_params, g)		
		# g.close()
		#Saving the shared model
		torch.save(model.state_dict(), os.path.join(model_path, "multihead_shared_model.pth"))
		# torch.save(model.state_dict(), os.path.join(head_path,  "multihead_model.pth"))
	else:
		#saving regularization parameters
		reg_params = model.reg_params
		f = open(os.path.join(model_path, "singlehead_reg_params.pickle"), 'wb')
		pickle.dump(reg_params, f)
		f.close()
		#Saving the shared model
		torch.save(model.state_dict(), os.path.join(model_path, "singlehead_shared_model.pth"))

	#save the performance of the model on the task to later determine the forgetting metric
	# with open(os.path.join(path_to_head, "performance.txt"), 'w') as file1:
	# 	input_to_txtfile = str(epoch_accuracy.item())
	# 	file1.write(input_to_txtfile)
	# 	file1.close()

def load_head(model, task_num, multi_head = True, use_gpu = False): #only used during the inference step of multi-head method. Single head doesn't need save/load.
	#make sure it is a multi_head approach
	print("Loading head (task#:{})".format(task_num))

	device = torch.device("cuda:0" if use_gpu else "cpu")
	if not multi_head:
		raise Exception("must be a multi_head approach")
	#Set paths
	model_path = os.path.join(os.getcwd(), "models")
	head_path  = os.path.join(model_path, "task_"+str(task_num))
	#Import model
	head_buffer = classification_head(model.tmodel.fc_head.in_features, model.tmodel.fc_head.out_features)
	head_buffer.load_state_dict(torch.load(os.path.join(head_path, "head.pth")))
	model.load_state_dict(torch.load(os.path.join(model_path, "multihead_shared_model.pth")))
	model.tmodel.fc_head.weight.data = head_buffer.fc.weight.data.to(device)
	model.tmodel.fc_head.bias.data = head_buffer.fc.bias.data.to(device)
	#use_gpu?
	# device = torch.device("cuda:0" if use_gpu else "cpu")
	# model.eval()
	# model.to(device)
	return model



def save_mask(model, task_num):
	#Create dir
	model_path = os.path.join(os.getcwd(), "models")
	head_path = os.path.join(model_path, "task_" + str(task_num)) #only for multi-head
	if not os.path.exists(head_path):
		os.mkdir(head_path)

	mask = []
	reg_params = model.reg_params
	for i, (name, params) in enumerate(model.tmodel.named_parameters()):
		mask.append(reg_params[i]['mask'])
	save_path = os.path.join(os.getcwd(), "models", "task_"+str(task_num), "mask"+str(task_num)+".pth")
	torch.save(mask, save_path)

def load_mask(model, task_num):
	reg_params = model.reg_params
	load_path = os.path.join(os.getcwd(), "models", "task_"+str(task_num), "mask"+str(task_num)+".pth")
	mask = torch.load(load_path)
	for i, (name, params) in enumerate(model.tmodel.named_parameters()):
		reg_params[i]['mask'] = mask[i]

def print_mask(model, task_list):
	mask = {}
	for task_num in task_list:
		print("*" * 100)
		print("Printing Trained Mask: {}".format(task_num))
		load_path = os.path.join(os.getcwd(), "models", "task_"+str(task_num), "mask"+str(task_num)+".pth")
		mask[task_num] = torch.load(load_path)
		for i, (name, params) in enumerate(model.tmodel.named_parameters()):
			if "weight" in name and "head" not in name:
				print(name)
				print("Nonzero count:", mask[task_num][i].nonzero().size(0), "/", mask[task_num][i].numel())
				print(mask[task_num][i])
	if len(task_list) == 2:  #compare two masks
		for i, (name, params) in enumerate(model.tmodel.named_parameters()):
			if "weight" in name and "head" not in name:
				a = mask[task_list[0]][i] * mask[task_list[1]][i]
				print("Overlap between mask_task{}_{} and mask_task{}_{} : ".format(task_list[0], name, task_list[1], name))
				print("\t", a.nonzero().size(0),"/",a.numel())
	p()


def print_network_param(model):
	print("*"*100)
	print("Printing Network Parameters")
	for i, (name, params) in enumerate(model.tmodel.named_parameters()):
		print(name)
		print(params)	


def save_result(accuracy, loaded_task, save_path, current_task):
	result_path = os.path.join(os.getcwd(), "results")
	result_file = result_path + "/"+save_path+"_load" + str(loaded_task) +  "_current" + str(current_task) +".pth"
	torch.save({
				'accuracy': accuracy,
				# 'epoch_loss': epoch_loss, 
				# 'epoch_accuracy': epoch_accuracy, 
				# 'model_state_dict': model_init.state_dict(),
				# 'optimizer_state_dict': optimizer.state_dict(),
				}, result_file)

def load_result(current_task, acc_list, save_path):
	result_path = os.path.join(os.getcwd(), "results")
	# acc = {}
	for loaded_task in range(current_task+1):
		result_file = result_path + "/"+save_path+"_load" + str(loaded_task) +  "_current" + str(current_task) +".pth"
		result = torch.load(result_file)
		accuracy = result['accuracy']
		if loaded_task not in acc_list:
			acc_list[loaded_task] = [accuracy]
		else:
			acc_list[loaded_task].append(accuracy)
		# print("loaded_head: {}, current_task: {}, accuracy: {:.4f}".format(loaded_task, current_task, accuracy))


def plot_result(task_num, save_path):
	acc_list ={}
	x_list = {}
	for i in range(task_num):
		load_result(i, acc_list, save_path)

	for i in range(task_num):
		for j, _ in enumerate(x_list):
			x_list[j].append(i)
		if i not in x_list:
			x_list[i] = [i]

	plt.figure()
	plt.xlabel('Sequence of Tasks')
	plt.ylabel('Accuracy')
	plt.title(save_path)
	for i in range(task_num):
		plt.plot(x_list[i], acc_list[i], label="Task {}".format(i))
	plt.legend()
	plt.savefig('./results/figures/'+save_path+'.jpg')
	plt.show()

def clear_models():
	model_path = os.path.join(os.getcwd(), "models")
	shutil.rmtree(model_path)
	# files = glob.glob(model_path, recursive = True)
	# for f in files:
	# 	os.rmdir(f)
	# for filename in os.listdir(model_path):
	# 	os.remove(model_path+ "/"+filename)