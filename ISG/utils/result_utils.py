import matplotlib.pyplot as plt
import sys
import os
import torch
import numpy as np
from datetime import date

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

	def to_csv(self):
		today = date.today()
		t = today.strftime("%m-%d-%Y")
		save_path = "./results/"+t+"/"
		if not os.path.isdir(save_path):
			os.makedirs(save_path)

		np.savetxt(save_path+self.args.method+"_SIM"+str(self.args.apply_SIM)+"_reg"+str(self.args.reglambda)+"_ACC.csv", np.asarray(self.ACC), delimiter=",")
		# np.savetxt(save_path+self.args.method+"_SIM"+str(self.args.apply_SIM)+"_reg"+str(self.args.reglambda)+"_LCA.csv", np.asarray(self.LCA), delimiter=",")
		np.savetxt(save_path+self.args.method+"_SIM"+str(self.args.apply_SIM)+"_reg"+str(self.args.reglambda)+"_FWT.csv", np.asarray(self.FWT), delimiter=",")
		np.savetxt(save_path+self.args.method+"_SIM"+str(self.args.apply_SIM)+"_reg"+str(self.args.reglambda)+"_BWT.csv", np.asarray(self.BWT), delimiter=",")
		np.savetxt(save_path+self.args.method+"_SIM"+str(self.args.apply_SIM)+"_reg"+str(self.args.reglambda)+"_SAT.csv", np.asarray(self.SAT), delimiter=",")
		np.savetxt(save_path+self.args.method+"_SIM"+str(self.args.apply_SIM)+"_reg"+str(self.args.reglambda)+"_PTB.csv", np.asarray(self.PTB), delimiter=",")
		np.savetxt(save_path+self.args.method+"_SIM"+str(self.args.apply_SIM)+"_reg"+str(self.args.reglambda)+"_IPK.csv", np.asarray(self.IPK), delimiter=",")


	def plot_results(self):
		return
