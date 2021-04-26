import matplotlib.pyplot as plt
import sys
import os
import glob
import torch
import numpy as np
from datetime import datetime
import re

def load_and_plot(task_num):
	result_path = os.path.join(os.getcwd(), "../results")
	y = {}
	for i in range(task_num+1):
		y[i] = []
	x = {}
	for i in range(task_num+1):
		x[i] = [u+i for u in range(task_num-i+1)]
	for current_task in range(task_num+1):
		for loaded_task in range(current_task, -1, -1):
			result_file = result_path + "/"+"load" + str(loaded_task) +  "_current" + str(current_task) +".pth"
			result = torch.load(result_file)
			accuracy = result['accuracy']
			y[loaded_task].append(accuracy)
			# print("loaded_head: {}, current_task: {}, accuracy: {:.4f}".format(loaded_task, current_task, accuracy))

	plt.title("permute_MNIST without ISG")
	plt.ylabel("Accuracy (%)")
	plt.xlabel("Number of tasks trained")
	plt.legend()
	for i in range(task_num+1):
		plt.plot(x[i], y[i])
	plt.legend(["task {}".format(i) for i in range(task_num)])
	plt.show()

#Imshow
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# img_grid =  torchvision.utils.make_grid(images)
# mat_imshow(img_grid, True)
# plt.show()

# load_and_plot(9)


class observer():
	#Analytic program that overlooks the entire simulation
	def __init__(self, args):
		self.ACC  = [] #[[ACC for 1st repetition], [ACC for 2nd repetition],...]
		self.LCA  = []
		self.FWT  = []
		self.BWT  = []
		self.SAT  = []
		self.PTB  = []
		self.IPK  = []
		self.args = args
		self.now  = datetime.now().strftime("%m-%d-%Y-%H:%M")

	def to_csv(self):
		t = self.now
		save_path = "./results/"+self.args.dataset+"_"+t+"/"
		if not os.path.isdir(save_path):
			os.makedirs(save_path)

		save_string = self.args.dataset+ \
					"_"+self.args.method+ \
					"_SIM"+str(self.args.apply_SIM)+ \
					"_r"+str(self.args.rho)+ \
					"_rand"+str(self.args.random_drop)+ \
					"_reg"+str(self.args.reglambda)+ \
					"_a"+str(self.args.alpha)+ \
					"_x"+str(self.args.xi)

		np.savetxt(save_path+save_string+"_ACC.csv", np.asarray(self.ACC), delimiter=",")
		# np.savetxt(save_path+save_string+"_LCA.csv", np.asarray(self.LCA), delimiter=",")
		np.savetxt(save_path+save_string+"_FWT.csv", np.asarray(self.FWT), delimiter=",")
		np.savetxt(save_path+save_string+"_BWT.csv", np.asarray(self.BWT), delimiter=",")
		np.savetxt(save_path+save_string+"_SAT.csv", np.asarray(self.SAT), delimiter=",")
		np.savetxt(save_path+save_string+"_PTB.csv", np.asarray(self.PTB), delimiter=",")
		np.savetxt(save_path+save_string+"_IPK.csv", np.asarray(self.IPK), delimiter=",")

	def legend_name(self,item):
		if "SIM1" in item and "rand1" not in item:
			return "SIM"
		elif "MAS" in item and "rand1" not in item:
			return "MAS"
		elif "SI_" in item:
			return "SI"
		elif "EWC" in item:
			return "EWC"
		elif "ANN" in item:
			return "ANN"
		elif "rand1" in item:
			return "RAND"

		# # else:
		# 	# return item[:3]
		return item[:-8]

	def plot_results(self):
		# Find n'th latest directory by modification date
		# result_path = sorted(glob.glob(os.path.join(os.getcwd(), 'results', '*/')), key=os.path.getmtime)[-1]
		result_path = "/home/sanghyun/Documents/CL/ISG/results/Plots_pMNIST/"
		# result_path = "/home/sanghyun/Documents/CL/ISG/results/Plots_sCIFAR/"
		print("PATH:    ", result_path)
		arr = os.listdir(result_path)
		c = {"SIM": 'r', "MAS": 'g', "SI": 'b', "EWC": 'y', "ANN":"k", "RAND":"c"}

		plt.figure()
		item_list=[]
		for item in arr:
			if 'ACC' in item:
				data = np.genfromtxt(result_path+item, delimiter=",")
				item_list.append(self.legend_name(item))
				plt.plot(np.arange(len(data)), data, color = c[item_list[-1]])

		plt.title("ACC")
		plt.legend(item_list)
		plt.xlabel("Task Sequence, t")
		plt.ylabel("ACC")
		plt.savefig(result_path+"plot_"+self.args.dataset+"_acc.jpg")
		# plt.show()
		
		plt.figure()
		item_list=[]
		for item in arr:
			if 'BWT' in item:
				data = np.genfromtxt(result_path+item, delimiter=",")
				item_list.append(self.legend_name(item))
				plt.plot(np.arange(len(data)), data, color = c[item_list[-1]])

		plt.title("BWT")
		plt.legend(item_list)
		plt.xlabel("Task, t")
		plt.ylabel("BWT")
		plt.savefig(result_path+"plot_"+self.args.dataset+"_bwt.jpg")
		# plt.show()
		
		plt.figure()
		item_list=[]
		for item in arr:
			if 'FWT' in item:
				data = np.genfromtxt(result_path+item, delimiter=",")
				item_list.append(self.legend_name(item))
				plt.plot(np.arange(len(data)), data, color = c[item_list[-1]])

		plt.title("FWT")
		plt.legend(item_list)
		plt.xlabel("Task Sequence, t")
		plt.ylabel("FWT")
		plt.savefig(result_path+"plot_"+self.args.dataset+"_fwt.jpg")
		# plt.show()

		plt.figure()
		item_list=[]
		for item in arr:
			if 'IPK' in item and "ANN" not in item:
				data = np.genfromtxt(result_path+item, delimiter=",")
				item_list.append(self.legend_name(item))
				plt.plot(np.arange(len(data)), data, color = c[item_list[-1]])

		plt.title("IPK")
		plt.legend(item_list)
		plt.xlabel("Task Sequence, t")
		plt.ylabel("IPK")
		plt.savefig(result_path+"plot_"+self.args.dataset+"_ipk.jpg")
		# plt.show()

		plt.figure()
		item_list=[]
		for item in arr:
			if 'PTB' in item and "ANN" not in item:
				data = np.genfromtxt(result_path+item, delimiter=",")
				item_list.append(self.legend_name(item))
				plt.plot(np.arange(len(data)), data, color = c[item_list[-1]])

		plt.title("PTB")
		plt.legend(item_list)
		plt.xlabel("Task Sequence, t")
		plt.ylabel("PTB")
		plt.savefig(result_path+"plot_"+self.args.dataset+"_ptb.jpg")
		# plt.show()

		plt.figure()
		item_list=[]
		for item in arr:
			if 'SAT' in item and "ANN" not in item:
				data = np.genfromtxt(result_path+item, delimiter=",")

				# #comment out below if not accumulating SAT
				# data_sum = 0
				# for i, d in enumerate(data):
				# 	data_sum += d
				# 	data[i] = data_sum

				item_list.append(self.legend_name(item))
				plt.plot(np.arange(len(data)), data, color = c[item_list[-1]])

		plt.title("SAT")
		plt.legend(item_list)
		plt.xlabel("Task Sequence, t")
		plt.ylabel("SAT")
		plt.savefig(result_path+"plot_"+self.args.dataset+"_sat.jpg")
		# plt.show()
	
# plot_results()

def pMNIST_weight_usage_plot():
	result_path = "/home/sanghyun/Documents/CL/ISG/outputs/pMNIST/04-16-21_13h-19m/"
	save_path = "/home/sanghyun/Documents/CL/ISG/results/Plots_pMNIST/"

	arr = os.listdir(result_path)
	for item in arr:
		if 'log' in item:
			f = open(result_path+item, "r")
			count = {}
			for line in f:
				if re.search(".weight",  line):
					layer = line[:3]
					if layer not in count:
						count[layer] = {}
				if re.search("tensor", line):
					a = line[9:-3].split(',')
					b = [int(num) for num in a]
					c = []
					for i in range(1, len(b)):
						c.append( (sum(b[i:])+(2000 - sum(b)))/2000  )
					for i in range(len(c)):
						if i not in count[layer]:
							count[layer][i] = [c[i]]
						else:
							count[layer][i].append(c[i])

	#Line Graph
	for layer, item in count.items():
		plt.figure()
		# plt.axis([1, 100, 0, 1])
		plt.title(r"Number of Tasks Encoded per Parameter, "+layer)
		plt.xlabel("Task Sequence, t")
		plt.ylabel('Parameter Count (%)')
		for key, data in item.items():
			plt.plot(np.arange(len(data)), data, label=r'$\geq$'+str(key+1)+' tasks')
		plt.legend()
		plt.savefig(save_path+"pMNIST_"+layer+"_Param_usage.jpg")

	#Bar Graph
	plt.figure()
	plt.title(r"Number of Tasks Encoded per Parameter")
	# plt.xlabel("Task Sequence, t")
	plt.ylabel('Parameter Count (%)')
	category = ['FC1', 'FC2']
	data_list = []
	for layer, item in count.items():
		# plt.axis([1, 100, 0, 1])
		print(layer)
		datas = []
		for key, data in item.items():
			datas.append(data[-1])
		data_list.append(datas)
	for i in range(10):
		plt.bar(np.arange(2)+0.07*i, [data_list[0][i], data_list[1][i]], 0.07,  label=r'$\geq$'+str(i+1)+' tasks')
		# plt.bar(np.arange(2)+0.04, data_list[1], 0.04, label = i)
	plt.xticks([0.3, 1.3], category)
	plt.legend()
	plt.savefig(save_path+"bar_pMNIST_"+layer+"_Param_usage.jpg")

def sCIFAR_weight_usage_plot():
	result_path = "/home/sanghyun/Documents/CL/ISG/outputs/sCIFAR/04-07-21_17:29/"
	save_path = "/home/sanghyun/Documents/CL/ISG/results/Plots_sCIFAR/"

	arr = os.listdir(result_path)
	for item in arr:
		if '55555' in item:
			f = open(result_path+item, "r")
			count = {}
			for line in f:
				if re.search(".weight",  line):
					layer = line[:5]
					if layer not in count:
						count[layer] = {}
				if re.search("tensor", line):
					a = line[9:-3].split(',')
					b = [int(num) for num in a]
					c = []
					# print(layer)
					# print("Layer :",layer," , b: ",b)
					# print(sum(b[1:])/sum(b))
					# p()
					for i in range(1, len(b)):
						c.append( sum(b[i:])/sum(b) )
					for i in range(len(c)):
						if i not in count[layer]:
							count[layer][i] = [c[i]]
						else:
							count[layer][i].append(c[i])

	#Line Graph
	for layer, item in count.items():
		plt.figure()
		# plt.axis([1, 10, 0, 1])
		plt.title(r"Number of Tasks Encoded per Parameter, "+layer)
		plt.xlabel("Task Sequence, t")
		plt.ylabel('Parameter Count (%)')
		for used_count, data in item.items():
			data.insert(0, 0)
			plt.plot(np.arange(len(data)), data, label=r'$\geq$'+str(used_count+1)+' tasks')
		plt.legend()
		plt.savefig(save_path+"sCIFAR_"+layer+"_Param_usage.jpg")
	# plt.show()

	#Bar Graph
	plt.figure()
	plt.title(r"Number of Tasks Encoded per Parameter")
	# plt.xlabel("Task Sequence, t")
	plt.ylabel('Parameter Count (%)')
	category = ['Conv.4', 'Dense']
	data_list = []
	for layer, item in count.items():
		# plt.axis([1, 100, 0, 1])
		print(layer)
		datas = []
		for key, data in item.items():
			datas.append(data[-1])
		data_list.append(datas)
	width = 0.09
	# data_list[3] = [0.999267578125, 0.948525390625, 0.90921875, 0.808857421875, 0.6401171875, 0.4079296875, 0.198212890625, 0.09419921875, 0.01056640625, 0.004765625]
	# data_list[4] = [1.0, 0.994140625, 0.96484375, 0.837890625, 0.595703125, 0.35546875, 0.16015625, 0.044921875, 0.0078125, 0.00390625]
	for i in range(10):
		plt.bar(np.arange(2)+width*i, [data_list[3][i], data_list[4][i]], width,  label=r'$\geq$'+str(i+1)+' tasks')
		# plt.bar(np.arange(2)+0.04, data_list[1], 0.04, label = i)
	plt.xticks([0.3, 1.3], category)
	plt.legend()
	plt.savefig(save_path+"bar_sCIFAR_"+layer+"_Param_usage.jpg")


# pMNIST_weight_usage_plot()
sCIFAR_weight_usage_plot()