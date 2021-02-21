from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


def f1(x, y):
    return np.sin(0.2*x + 0.1) ** 4 + np.cos(10 + 0.2*y * x) * np.cos(y) - 0.2*y + 0.1 * x
def f2(x, y):
    return -np.cos(y + 0.1) - np.sin(2 + 0.5*x * 0.5*y) 

def g1(x):
	return np.sin(x) + 1.4 * np.sin(0.8*x + 0.1) + 0.5 * np.sin(3*x + 0.2) - 0.2*x

def g2(x):
	return np.sin(0.4*x)**3 + 1.7 * np.sin(0.5*x + 0.2)**2 + 0.4 * np.sin(2*x + 0.3) + 0.1*x


def draw_contour_f1():
	x = np.linspace(3, 8, 70)  #f1map
	y = np.linspace(5, 8, 70)  #f1map
	X, Y = np.meshgrid(x, y)
	Z = f1(X, Y)

	fig = plt.figure()
	plot_3d = False
	if plot_3d:
		ax = plt.axes(projection = '3d')
		ax.contour3D(X, Y, Z, 80, cmap='RdYlGn')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_zticklabels([])
		ax.set_xlabel(r'$\theta_1$')
		ax.set_ylabel(r'$\theta_2$')
		ax.set_zlabel(r'$\theta_3$')
	else:
		plt.figure(figsize=(5,5))
		ax = plt.axes()
		ax.contour(X, Y, Z, 80, cmap='RdYlGn')
		ax.set_xlabel(r'$\theta_1$', fontsize = 18)
		ax.set_ylabel(r'$\theta_2$', fontsize = 18)
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		# ax.set_xticks([])
		# ax.set_yticks([])
		# plt.contour(X, Y, Z, 40, colors = 'RdYlGn')
		# plt.colorbar();
	plt.savefig('../figures/f1map.jpg')
	# plt.show()

def draw_contour_f2():
	x = np.linspace(2, 6, 70)  #f2map
	y = np.linspace(2, 7, 70)  #f2map
	X, Y = np.meshgrid(x, y)
	Z = f2(X, Y)

	fig = plt.figure()
	plot_3d = False
	if plot_3d:
		ax = plt.axes(projection = '3d')
		ax.contour3D(X, Y, Z, 80, cmap='RdYlGn')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_zticklabels([])
		ax.set_xlabel(r'$\theta_1$')
		ax.set_ylabel(r'$\theta_2$')
		ax.set_zlabel(r'$\theta_3$')
	else:
		plt.figure(figsize=(5,5))
		# ax = plt.axes()
		plt.contour(X, Y, Z, 80, cmap='RdYlGn')
		# plt.colorbar()
		ax = plt.axes()
		ax.set_xlabel(r'$\theta_1$', fontsize = 18)
		ax.set_ylabel(r'$\theta_2$', fontsize = 18)
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		# ax.set_xticks([])
		# ax.set_yticks([])
	plt.savefig('../figures/f2map.jpg')
	# plt.show()

def draw_subspace1():
	x = np.linspace(0, 10, 70)
	z = g1(x)
	plt.figure(figsize=(5,5))
	ax = plt.axes()
	ax.plot(x, z, color='black')
	ax.set_xlabel(r'$\theta_g$', fontsize = 18)
	ax.set_ylabel(r'$MSE_{g(x_k)}$', fontsize = 18)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	plt.savefig('../figures/gmap1.jpg')
	# plt.show()

def draw_subspace2():
	x = np.linspace(0, 10, 70)
	z = g2(x)
	plt.figure(figsize=(5,5))
	ax = plt.axes()
	ax.plot(x, z, color='black')
	ax.set_xlabel(r'$\theta_h$', fontsize = 18)
	ax.set_ylabel(r'$MSE_{h(x_{k+1})}$', fontsize = 18)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	plt.savefig('../figures/gmap2.jpg')
	# plt.show()


draw_subspace1()
draw_subspace2()
draw_contour_f1()
draw_contour_f2()