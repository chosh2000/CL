import matplotlib.pyplot as plt


filename1 = open('/home/sanghyun/Documents/CL/survey/CLsurvey_hsu/outputs/permuted_MNIST_incremental_task/MAS/12_02_00h_47m/MAS.log', 'r')
filename2 = open('/home/sanghyun/Documents/CL/survey/CLsurvey_hsu/outputs/permuted_MNIST_incremental_task/SI/12_02_00h_47m/SI.log', 'r')
filename3 = open('/home/sanghyun/Documents/CL/survey/CLsurvey_hsu/outputs/permuted_MNIST_incremental_task/L2/12_02_00h_47m/L2.log', 'r')


def acc_list(filename):
	acc = []
	data = filename.readlines()
	for line in data:
		if 'average acc' in line:
			i = line.index(':')
			acc.append(float(line[i+2:i+6]))
	return acc

acc_MAS= acc_list(filename1)
acc_SI= acc_list(filename2)
acc_L2= acc_list(filename3)


plt.figure()
plt.plot(range(100), acc_MAS, label='MAS')
plt.plot(range(100), acc_SI, label='SI')
plt.plot(range(100), acc_L2, label='L2')
plt.legend()
plt.show()