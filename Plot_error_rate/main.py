from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from skimage.transform import resize
from matplotlib import offsetbox
import pandas as pd
import scipy.io
import csv
import scipy.misc
import os
import math
from sklearn.model_selection import train_test_split   #--> https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


def main():
    method = "RDA"  #--> RDA, kernel_RDA
    plot_error_MNIST_in_RDA(method=method, max_n_componenets=10, show_legends=True)

def plot_error_MNIST_in_RDA(method="RDA", max_n_componenets=30, show_legends=True):
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ["r", "b", "g", "darkorchid", "darkorange"]
    markers = ["o", "s", "^", "v", ">"]
    labels = ["r1=0, r2=0", "r1=0, r2=1", "r1=1, r2=0", "r1=1, r2=1", "r1=0.5, r2=0.5"]
    counter = -1
    for r_1 in [0, 1]:
        for r_2 in [0, 1]:
            counter = counter + 1
            path = './MNIST_rates/' + method + '/' + "r_1=" + str(r_1) + ", r_2=" + str(r_2) + "/"
            error = load_variable(name_of_variable="error", path=path)
            error = error[0:max_n_componenets]
            n_comonent = np.arange(1, len(error)+1, 1)
            ax.plot(n_comonent, error, marker=markers[counter], color=colors[counter], markersize=10, label=labels[counter], fillstyle='none')  #--> fillstyle = ('full', 'left', 'right', 'bottom', 'top', 'none')
    r_1, r_2 = 0.5, 0.5
    path = './MNIST_rates/' + method + '/' + "r_1=" + str(r_1) + ", r_2=" + str(r_2) + "/"
    error = load_variable(name_of_variable="error", path=path)
    error = error[0:max_n_componenets]
    n_comonent = np.arange(1, len(error)+1, 1)
    ax.plot(n_comonent, error, marker=markers[counter+1], color=colors[counter+1], markersize=10, label=labels[counter+1], fillstyle='none')
    # plt.grid(color='k', linestyle='-', linewidth=2)
    plt.grid()
    plt.xlabel("Dimensionality of subspace", fontsize=13)
    plt.ylabel("Error", fontsize=13)
    plt.xticks(n_comonent)
    if show_legends is not None:
        ax.legend()
    plt.show()

def load_variable(name_of_variable, path='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    file_address = path + name_of_variable + '.pckl'
    f = open(file_address, 'rb')
    variable = pickle.load(f)
    f.close()
    return variable

if __name__ == '__main__':
    main()