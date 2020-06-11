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
    plot_supervisionLevel_heatmap_2()
    plot_supervisionLevel_plane()

def plot_supervisionLevel_plane():
    # https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    y, x = np.meshgrid(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000))
    z = (x + y) / 2
    surf = ax.plot_surface(x, y, z, cmap='RdBu', linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def plot_supervisionLevel_heatmap_2():
    # https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib
    y, x = np.meshgrid(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000))
    # z = (1 - x / 2. + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
    z = (x + y) / 2
    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    z = z[:-1, :-1]
    z_min, z_max = z.min(), z.max()
    fig, ax = plt.subplots()
    c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)  # colormaps: 'RdBu', cm.jet, 'gray', cm.coolwarm
    # ax.set_title('pcolormesh')
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plt.show()

def plot_supervisionLevel_heatmap_1():
    # https://stackoverflow.com/questions/2369492/generate-a-heatmap-in-matplotlib-using-a-scatter-data-set
    # Generate some test data
    x = np.random.randn(100)
    y = np.random.randn(100)
    s = 60
    img, extent = my_heatmap(x, y, s)
    heatmap = plt.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
    plt.colorbar(heatmap)
    plt.show()

def my_heatmap(x, y, s, bins=1000):
    # https://stackoverflow.com/questions/2369492/generate-a-heatmap-in-matplotlib-using-a-scatter-data-set
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

if __name__ == '__main__':
    main()