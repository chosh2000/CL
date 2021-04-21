import matplotlib.pyplot as plt
import sys
import os
import glob
import torch
import numpy as np
from datetime import datetime

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
