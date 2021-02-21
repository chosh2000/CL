import matplotlib.pyplot as plt
import numpy as np
import random

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    alpha_list = {} #this is for r_{ij}
    alpha_list2 = {} #this is for \Omega_{ij}
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # White background Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4, fill = True)
            ax.add_artist(circle)
    # color nodes
    for n, layer_size in enumerate(layer_sizes):
        alpha_list[n] = np.random.random(layer_size)
        alpha_list2[n] = np.random.random(layer_size)
        if layer_size == 1:
            alpha_list[n][0] = 1
            alpha_list2[n][0] = 0
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='k', ec='k', zorder=4, fill = True, alpha = alpha_list[n][m])
            ax.add_artist(circle)

    # r_{ij} Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                h_i = alpha_list[n][m]
                h_j = alpha_list[n+1][o]
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k', alpha = (h_i * h_j)**0.8, lw=2*(h_i * h_j)**0.2)
                ax.add_artist(line)

    # # \Omega_{ij} Edges
    # for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
    #     layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
    #     layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
    #     for m in range(layer_size_a):
    #         for o in range(layer_size_b):
    #             h_i = alpha_list2[n][m]
    #             h_j = alpha_list2[n+1][o]
    #             line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
    #                               [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='r', alpha = (h_i * h_j)**0.8, lw=2*(h_i * h_j)**0.2)
    #             ax.add_artist(line)

np.random.seed(7)
fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
ax.axis('off')
draw_neural_net(ax, .1, .9, .1, .9, [5, 5, 5, 1])
fig.savefig('../figures/NN_diagram/nn.png')