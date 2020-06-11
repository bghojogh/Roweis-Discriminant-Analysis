from my_RDA import My_RDA
from my_kernel_RDA import My_kernel_RDA
from my_FDA import My_FDA
from my_kernel_FDA import My_kernel_FDA
from my_kernel_PCA import My_kernel_PCA
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
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing   # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
from sklearn.model_selection import KFold


def main():
    # ---- settings:
    dataset = "ATT_glasses"  #--> ATT_glasses, Binary_XOR, Concentric_rings, MNIST,
                                      # regression_bechmark1, regression_bechmark2, regression_bechmark3, Facebook, forest_fire
    task = "classification"  #--> classification, regression
    manifold_learning_method = "FDA" #--> RDA, kernel_RDA, FDA, kernel_FDA, kernel_PCA
    kernel_on_X = "rbf"  #kernel over data (X) --> ‘rbf’, ‘sigmoid’, ‘polynomial’, ‘poly’, ‘linear’, ‘cosine’ --> if None, it is linear
    if task == "classification":
        kernel_on_labels = "delta_kernel"  # kernel over labels (Y) --> 'delta_kernel', ‘rbf’, ‘sigmoid’, ‘polynomial’, ‘poly’, ‘linear’, ‘cosine’ --> if None, it is linear
    elif task == "regression":
        kernel_on_labels = "rbf"
    r_1 = 0
    r_2 = 0
    generate_dataset_again = False
    split_to_train_and_test = True
    save_projection_directions_again = False
    reconstruct_using_howMany_projection_directions = None  # --> an integer >= 1, if None: using all "specified" directions when creating the python class
    process_out_of_sample_all_together = True
    project_out_of_sample = True
    n_projection_directions_to_save = 20 #--> an integer >= 1, if None: save all "specified" directions when creating the python class
    save_reconstructed_images_again = False
    save_reconstructed_outOfSample_images_again = False
    if dataset == "ATT_glasses":
        indices_reconstructed_images_to_save = None  #--> [100, 120]
        outOfSample_indices_reconstructed_images_to_save = None  #--> [100, 120]
    plot_projected_pointsAndImages_again = True
    which_dimensions_to_plot_inpointsAndImagesPlot = [0,1] #--> list of two indices (start and end), e.g. [1,3] or [0,1]
    subset_of_MNIST = True
    pick_subset_of_MNIST_again = False
    MNIST_subset_cardinality_training = 5000
    MNIST_subset_cardinality_testing = 1000

    if dataset == "ATT_glasses":
        path_dataset = "./Att_glasses/"
        n_samples = 400
        image_height = 112
        image_width = 92
        data = np.zeros((image_height * image_width, n_samples))
        labels = np.zeros((1, n_samples))
        image_index = -1
        for class_index in range(2):
            for filename in os.listdir(path_dataset + "class" + str(class_index+1) + "/"):
                image_index = image_index + 1
                if image_index >= n_samples:
                    break
                img = load_image(address_image=path_dataset + "class" + str(class_index+1) + "/" + filename)
                data[:, image_index] = img.ravel()
                labels[:, image_index] = class_index
        # ---- cast dataset from string to float:
        data = data.astype(np.float)
        # ---- cnormalize (standardation):
        data_notNormalized = data
        # data = data / 255
        scaler = StandardScaler(with_mean=True, with_std=True).fit(data.T)
        data = (scaler.transform(data.T)).T
        X_train = data
        Y_train = labels
    elif dataset == "Binary_XOR":
        if generate_dataset_again:
            XOR_radius = 2
            x1 = np.random.multivariate_normal(mean=(XOR_radius, XOR_radius), cov=[[1, 0], [0, 1]], size=100)
            x2 = np.random.multivariate_normal(mean=(-XOR_radius, -XOR_radius), cov=[[1, 0], [0, 1]], size=100)
            x3 = np.random.multivariate_normal(mean=(-XOR_radius, XOR_radius), cov=[[1, 0], [0, 1]], size=100)
            x4 = np.random.multivariate_normal(mean=(XOR_radius, -XOR_radius), cov=[[1, 0], [0, 1]], size=100)
            data_XOR = np.vstack((x1, x2, x3, x4)).T
            # data_appended = np.random.rand(8, 400)
            # data = np.vstack((data_XOR, data_appended))
            data = data_XOR
            labels = [0] * 200
            labels.extend([1] * 200)
            labels = np.asarray(labels)
            # scaler = StandardScaler(with_mean=True, with_std=True).fit(data.T)
            # data = (scaler.transform(data.T)).T
            save_variable(variable=data, name_of_variable="data", path_to_save='./Binary_XOR/')
            save_variable(variable=labels, name_of_variable="labels", path_to_save='./Binary_XOR/')
            if split_to_train_and_test:
                X_train, X_test, Y_train, Y_test = train_test_split(data.T, labels, test_size=0.3, random_state=42)
                X_train = X_train.T
                X_test = X_test.T
                save_variable(variable=X_train, name_of_variable="X_train", path_to_save='./Binary_XOR/')
                save_variable(variable=X_test, name_of_variable="X_test", path_to_save='./Binary_XOR/')
                save_variable(variable=Y_train, name_of_variable="Y_train", path_to_save='./Binary_XOR/')
                save_variable(variable=Y_test, name_of_variable="Y_test", path_to_save='./Binary_XOR/')
                # class_legends = ["class 1", "class 2"]
                class_legends = None
                plot_components_by_colors_withTest(X_train=X_train, X_test=X_test, y_train=Y_train, y_test=Y_test, which_dimensions_to_plot=[0,1], class_legends=class_legends, markersize=8, colors=["b", "r"], markers=["s", "o"])
            else:
                X_train = data
                Y_train = labels
                save_variable(variable=X_train, name_of_variable="X_train", path_to_save='./Binary_XOR/')
                save_variable(variable=Y_train, name_of_variable="Y_train", path_to_save='./Binary_XOR/')
        else:
            if split_to_train_and_test:
                X_train = load_variable(name_of_variable="X_train", path='./Binary_XOR/')
                X_test = load_variable(name_of_variable="X_test", path='./Binary_XOR/')
                Y_train = load_variable(name_of_variable="Y_train", path='./Binary_XOR/')
                Y_test = load_variable(name_of_variable="Y_test", path='./Binary_XOR/')
            else:
                X_train = load_variable(name_of_variable="X_train", path='./Binary_XOR/')
                Y_train = load_variable(name_of_variable="Y_train", path='./Binary_XOR/')
    elif dataset == "Concentric_rings":
        if generate_dataset_again:
            ring1_radius = 2
            ring2_radius = 5
            data = np.zeros((2, 400))
            for sample_index in range(200):
                angle = np.random.uniform(0, 2 * math.pi)  # in radians
                distance = math.sqrt(np.random.uniform(0, ring1_radius))
                point = np.array([distance * math.cos(angle), distance * math.sin(angle)]).reshape((-1, 1))
                data[:, sample_index] = point.ravel()
            for sample_index in range(201, 400):
                angle = np.random.uniform(0, 2 * math.pi)  # in radians
                distance = math.sqrt(np.random.uniform(ring1_radius, ring2_radius))
                point = np.array([distance * math.cos(angle), distance * math.sin(angle)]).reshape((-1, 1))
                data[:, sample_index] = point.ravel()
            labels = [0] * 200
            labels.extend([1] * 200)
            labels = np.asarray(labels)
            # scaler = StandardScaler(with_mean=True, with_std=True).fit(data.T)
            # data = (scaler.transform(data.T)).T
            save_variable(variable=data, name_of_variable="data", path_to_save='./Concentric_rings/')
            save_variable(variable=labels, name_of_variable="labels", path_to_save='./Concentric_rings/')
            if split_to_train_and_test:
                X_train, X_test, Y_train, Y_test = train_test_split(data.T, labels, test_size=0.3, random_state=42)
                X_train = X_train.T
                X_test = X_test.T
                save_variable(variable=X_train, name_of_variable="X_train", path_to_save='./Concentric_rings/')
                save_variable(variable=X_test, name_of_variable="X_test", path_to_save='./Concentric_rings/')
                save_variable(variable=Y_train, name_of_variable="Y_train", path_to_save='./Concentric_rings/')
                save_variable(variable=Y_test, name_of_variable="Y_test", path_to_save='./Concentric_rings/')
                # class_legends = ["class 1", "class 2"]
                class_legends = None
                plot_components_by_colors_withTest(X_train=X_train, X_test=X_test, y_train=Y_train, y_test=Y_test, which_dimensions_to_plot=[0,1], class_legends=class_legends, markersize=8, colors=["b", "r"], markers=["s", "o"])
            else:
                X_train = data
                Y_train = labels
                save_variable(variable=X_train, name_of_variable="X_train", path_to_save='./Concentric_rings/')
                save_variable(variable=Y_train, name_of_variable="Y_train", path_to_save='./Concentric_rings/')
        else:
            if split_to_train_and_test:
                X_train = load_variable(name_of_variable="X_train", path='./Concentric_rings/')
                X_test = load_variable(name_of_variable="X_test", path='./Concentric_rings/')
                Y_train = load_variable(name_of_variable="Y_train", path='./Concentric_rings/')
                Y_test = load_variable(name_of_variable="Y_test", path='./Concentric_rings/')
            else:
                X_train = load_variable(name_of_variable="X_train", path='./Concentric_rings/')
                Y_train = load_variable(name_of_variable="Y_train", path='./Concentric_rings/')
    elif dataset == 'MNIST':
        path_dataset_save = "./MNIST/"
        file = open(path_dataset_save+'X_train.pckl','rb')
        X_train = pickle.load(file); file.close()
        file = open(path_dataset_save+'y_train.pckl','rb')
        Y_train = pickle.load(file); file.close()
        Y_train = np.asarray(Y_train)
        file = open(path_dataset_save+'X_test.pckl','rb')
        X_test = pickle.load(file); file.close()
        file = open(path_dataset_save+'y_test.pckl','rb')
        Y_test = pickle.load(file); file.close()
        Y_test = np.asarray(Y_test)
        if subset_of_MNIST:
            if pick_subset_of_MNIST_again:
                X_train_picked = X_train[0:MNIST_subset_cardinality_training, :]
                X_test_picked = X_test[0:MNIST_subset_cardinality_testing, :]
                y_train_picked = Y_train[0:MNIST_subset_cardinality_training]
                y_test_picked = Y_test[0:MNIST_subset_cardinality_testing]
                save_variable(X_train_picked, 'X_train_picked', path_to_save=path_dataset_save)
                save_variable(X_test_picked, 'X_test_picked', path_to_save=path_dataset_save)
                save_variable(y_train_picked, 'y_train_picked', path_to_save=path_dataset_save)
                save_variable(y_test_picked, 'y_test_picked', path_to_save=path_dataset_save)
            else:
                file = open(path_dataset_save + 'X_train_picked.pckl', 'rb')
                X_train_picked = pickle.load(file)
                file.close()
                file = open(path_dataset_save + 'X_test_picked.pckl', 'rb')
                X_test_picked = pickle.load(file)
                file.close()
                file = open(path_dataset_save + 'y_train_picked.pckl', 'rb')
                y_train_picked = pickle.load(file)
                file.close()
                file = open(path_dataset_save + 'y_test_picked.pckl', 'rb')
                y_test_picked = pickle.load(file)
                file.close()
            X_train = X_train_picked
            X_test = X_test_picked
            Y_train = y_train_picked
            Y_test = y_test_picked
        X_train = X_train.T
        X_test = X_test.T
        scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train.T)
        X_train = (scaler.transform(X_train.T)).T
        X_test = (scaler.transform(X_test.T)).T
    elif dataset == "regression_bechmark1" or dataset == "regression_bechmark2" or dataset == "regression_bechmark3":
        if generate_dataset_again:
            n_datasets = 50
            n_samples = 100
            if dataset == "regression_bechmark1":
                dimensionality = 4
            elif dataset == "regression_bechmark2":
                dimensionality = 4
            elif dataset == "regression_bechmark3":
                dimensionality = 10
            X_train_stacked = np.zeros((dimensionality, int(n_samples * 0.7), n_datasets))
            X_test_stacked = np.zeros((dimensionality, int(n_samples * 0.3), n_datasets))
            Y_train_stacked = np.zeros((1, int(n_samples * 0.7), n_datasets))
            Y_test_stacked = np.zeros((1, int(n_samples * 0.3), n_datasets))
            for dataset_index in range(n_datasets):
                if dataset == "regression_bechmark1":
                    mean = np.zeros((dimensionality,))
                    cov = np.eye(dimensionality)
                    data = np.random.multivariate_normal(mean=mean, cov=cov, size=n_samples)
                    data = data.T
                    epsilon = np.random.normal(loc=0, scale=1, size=1)
                    labels = (data[0,:] / (0.5 + (data[1,:]+1.5)**2)) + (1 + data[1,:])**2 + (0.5 * epsilon)
                elif dataset == "regression_bechmark2":
                    data = np.random.uniform(low=0.7001, high=1, size=(dimensionality, n_samples))
                    epsilon = np.random.normal(loc=0, scale=1, size=1)
                    labels = (np.sin((math.pi * data[1,:]) + 1))**2 + (0.5 * epsilon)
                elif dataset == "regression_bechmark3":
                    mean = np.zeros((dimensionality,))
                    cov = np.eye(dimensionality)
                    data = np.random.multivariate_normal(mean=mean, cov=cov, size=n_samples)
                    data = data.T
                    epsilon = np.random.normal(loc=0, scale=1, size=1)
                    labels = 0.5 * (data[0, :] ** 2) * epsilon
                X_train, X_test, Y_train, Y_test = train_test_split(data.T, labels, test_size=0.3, random_state=42)
                X_train = X_train.T
                X_test = X_test.T
                X_train_stacked[:, :, dataset_index] = X_train
                X_test_stacked[:, :, dataset_index] = X_test
                Y_train_stacked[:, :, dataset_index] = Y_train
                Y_test_stacked[:, :, dataset_index] = Y_test
            path = "./Regression_bechmarks/" + dataset + "/"
            save_variable(X_train_stacked, 'X_train_stacked', path_to_save=path)
            save_variable(X_test_stacked, 'X_test_stacked', path_to_save=path)
            save_variable(Y_train_stacked, 'Y_train_stacked', path_to_save=path)
            save_variable(Y_test_stacked, 'Y_test_stacked', path_to_save=path)
        else:
            n_datasets = 50
            path = "./Regression_bechmarks/" + dataset + "/"
            X_train_stacked = load_variable(name_of_variable="X_train_stacked", path=path)
            X_test_stacked = load_variable(name_of_variable="X_test_stacked", path=path)
            Y_train_stacked = load_variable(name_of_variable="Y_train_stacked", path=path)
            Y_test_stacked = load_variable(name_of_variable="Y_test_stacked", path=path)
    elif dataset == "Facebook":
        path_dataset = './facebook_dataset/dataset_Facebook.csv'
        data, labels = read_facebook_dataset(path_dataset, sep=";", header='infer')
        scaler = StandardScaler(with_mean=True, with_std=True).fit(data.T)
        data = (scaler.transform(data.T)).T
    elif dataset == 'forest_fire':
        path_dataset = './forest_fire_dataset/forestfires.csv'
        data, labels = read_regression_dataset_withOneOutput(path_dataset, column_of_labels=-1, sep=",", header='infer')
        labels = labels.reshape((1, -1))
        scaler = StandardScaler(with_mean=True, with_std=True).fit(data.T)
        data = (scaler.transform(data.T)).T

    # ---- for regression benchmark datasets:
    if dataset == "regression_bechmark1" or dataset == "regression_bechmark2" or dataset == "regression_bechmark3":
        n_components = 2
        error = np.zeros((n_datasets,))
        for dataset_index in range(n_datasets):
            #--- fit the subspace learning:
            if manifold_learning_method == "RDA":
                my_manifold_learning = My_RDA(r_1=r_1, r_2=r_2, n_components=n_components, kernel_on_labels=kernel_on_labels)
            elif manifold_learning_method == "kernel_RDA":
                my_manifold_learning = My_kernel_RDA(r_1=r_1, r_2=r_2, n_components=n_components, kernel_on_X=kernel_on_X, kernel_on_labels=kernel_on_labels)
            data_transformed = my_manifold_learning.fit_transform(X=X_train_stacked[:, :, dataset_index], Y=Y_train_stacked[:, :, dataset_index])
            # ---- transform out-of-sample data:
            data_outOfSample_transformed = my_manifold_learning.transform(X=X_test_stacked[:, :, dataset_index], Y=Y_test_stacked[:, :, dataset_index])
            #--- regression:
            reg = LinearRegression().fit(X=data_transformed.T, y=Y_train_stacked[:, :, dataset_index].ravel())
            y_pred = reg.predict(X=data_outOfSample_transformed.T)
            error[dataset_index] = np.sqrt(metrics.mean_squared_error(y_true=Y_test_stacked[:, :, dataset_index].ravel(), y_pred=y_pred))
        error_mean = error.mean()
        error_std = error.std()
        print("RMSE: ", str(error_mean) + " +- " + str(error_std))
        path = "./output/"+dataset+"/"+manifold_learning_method+"/r_1="+str(r_1)+", r_2="+str(r_2)+"/"
        save_variable(error_mean, 'error_mean', path_to_save=path)
        save_variable(error_std, 'error_std', path_to_save=path)
        save_np_array_to_txt(variable=error_mean, name_of_variable="error_mean", path_to_save=path)
        save_np_array_to_txt(variable=error_std, name_of_variable="error_std", path_to_save=path)
        # --- regression on not-transformed dataset:
        for dataset_index in range(n_datasets):
            #--- regression:
            reg = LinearRegression().fit(X=X_train_stacked[:, :, dataset_index].T, y=Y_train_stacked[:, :, dataset_index].ravel())
            y_pred = reg.predict(X=X_test_stacked[:, :, dataset_index].T)
            error[dataset_index] = np.sqrt(metrics.mean_squared_error(y_true=Y_test_stacked[:, :, dataset_index].ravel(), y_pred=y_pred))
        error_mean = error.mean()
        error_std = error.std()
        print("RMSE for original data: ", str(error_mean) + " +- " + str(error_std))
        path = "./output/" + dataset + "/" + manifold_learning_method + "/original_data/"
        save_variable(error_mean, 'error_mean', path_to_save=path)
        save_variable(error_std, 'error_std', path_to_save=path)
        save_np_array_to_txt(variable=error_mean, name_of_variable="error_mean", path_to_save=path)
        save_np_array_to_txt(variable=error_std, name_of_variable="error_std", path_to_save=path)
        return

    # regression:
    if dataset == "Facebook" or dataset == "forest_fire":
        n_components = 2
        data_T = data.T
        labels_T = labels.T
        kf = KFold(n_splits=10)
        error = np.zeros((10,))
        fold_index = -1
        for train_index, test_index in kf.split(data.T):
            fold_index = fold_index + 1
            X_train, X_test = data_T[train_index].T, data_T[test_index].T
            Y_train, Y_test = labels_T[train_index].T, labels_T[test_index].T
            #--- fit the subspace learning:
            if manifold_learning_method == "RDA":
                my_manifold_learning = My_RDA(r_1=r_1, r_2=r_2, n_components=n_components, kernel_on_labels=kernel_on_labels)
            elif manifold_learning_method == "kernel_RDA":
                my_manifold_learning = My_kernel_RDA(r_1=r_1, r_2=r_2, n_components=n_components, kernel_on_X=kernel_on_X, kernel_on_labels=kernel_on_labels)
            data_transformed = my_manifold_learning.fit_transform(X=X_train, Y=Y_train)
            # ---- transform out-of-sample data:
            data_outOfSample_transformed = my_manifold_learning.transform(X=X_test, Y=Y_test)
            #--- regression:
            reg = LinearRegression().fit(X=data_transformed.T, y=Y_train.T)
            y_pred = reg.predict(X=data_outOfSample_transformed.T)
            error[fold_index] = np.sqrt(metrics.mean_squared_error(y_true=Y_test.T, y_pred=y_pred))
        error_mean = error.mean()
        error_std = error.std()
        print("RMSE: ", str(error_mean) + " +- " + str(error_std))
        path = "./output/"+dataset+"/"+manifold_learning_method+"/r_1="+str(r_1)+", r_2="+str(r_2)+"/"
        save_variable(error_mean, 'error_mean', path_to_save=path)
        save_variable(error_std, 'error_std', path_to_save=path)
        save_np_array_to_txt(variable=error_mean, name_of_variable="error_mean", path_to_save=path)
        save_np_array_to_txt(variable=error_std, name_of_variable="error_std", path_to_save=path)
        return

    # ---- fit + transform training data:
    if manifold_learning_method == "RDA":
        my_manifold_learning = My_RDA(r_1=r_1, r_2=r_2, n_components=None, kernel_on_labels=kernel_on_labels)
        data_transformed = my_manifold_learning.fit_transform(X=X_train, Y=Y_train)
    elif manifold_learning_method == "kernel_RDA":
        my_manifold_learning = My_kernel_RDA(r_1=r_1, r_2=r_2, n_components=None, kernel_on_X=kernel_on_X, kernel_on_labels=kernel_on_labels)
        data_transformed = my_manifold_learning.fit_transform(X=X_train, Y=Y_train)
    elif manifold_learning_method == "FDA":
        my_manifold_learning = My_FDA(n_components=None, kernel=kernel_on_X)
        data_transformed = my_manifold_learning.fit_transform(X=X_train, y=Y_train)
    elif manifold_learning_method == "kernel_FDA":
        my_manifold_learning = My_kernel_FDA(n_components=None, kernel=kernel_on_X)
        data_transformed = my_manifold_learning.fit_transform(X=X_train, y=Y_train)
    elif manifold_learning_method == "kernel_PCA":
        my_manifold_learning = My_kernel_PCA(n_components=None, kernel=kernel_on_X)
        data_transformed = my_manifold_learning.fit_transform(X=X_train)

    # ---- transform out-of-sample data:
    if project_out_of_sample:
        if manifold_learning_method == "RDA" or manifold_learning_method == "kernel_RDA":
            data_outOfSample_transformed = my_manifold_learning.transform(X=X_test, Y=Y_test)

    # ---- save projection directions:
    if save_projection_directions_again:
        print("Saving projection directions...")
        projection_directions = my_manifold_learning.get_projection_directions()
        if n_projection_directions_to_save == None:
            n_projection_directions_to_save = projection_directions.shape[1]
        for projection_direction_index in range(n_projection_directions_to_save):
            an_image = projection_directions[:, projection_direction_index].reshape((image_height, image_width))
            # scale (resize) image array:
            an_image = scipy.misc.imresize(arr=an_image, size=500)  # --> 5 times bigger
            # save image:
            if manifold_learning_method == "RDA" or manifold_learning_method == "kernel_RDA":
                save_image(image_array=an_image, path_without_file_name="./output/"+dataset+"/"+manifold_learning_method+"/r_1="+str(r_1)+", r_2="+str(r_2)+"/directions/", file_name=str(projection_direction_index)+".png")
            else:
                save_image(image_array=an_image, path_without_file_name="./output/"+dataset+"/"+manifold_learning_method+"/directions/", file_name=str(projection_direction_index)+".png")

    # ---- save reconstructed images:
    if save_reconstructed_images_again:
        X_reconstructed = my_manifold_learning.reconstruct(X=X_train, scaler=scaler, using_howMany_projection_directions=reconstruct_using_howMany_projection_directions)
        if indices_reconstructed_images_to_save == None:
            indices_reconstructed_images_to_save = [0, X_reconstructed.shape[1]]
        for image_index in range(indices_reconstructed_images_to_save[0], indices_reconstructed_images_to_save[1]):
            an_image = X_reconstructed[:, image_index].reshape((image_height, image_width))
            # scale (resize) image array:
            an_image = scipy.misc.imresize(arr=an_image, size=500)  # --> 5 times bigger
            # save image:
            if reconstruct_using_howMany_projection_directions is not None:
                tmp = "_using" + str(reconstruct_using_howMany_projection_directions) + "Directions"
            else:
                tmp = "_usingAllDirections"
            save_image(image_array=an_image, path_without_file_name="./output/"+dataset+"/"+manifold_learning_method+"/r_1="+str(r_1)+", r_2="+str(r_2)+"/reconstructed_train"+tmp+"/", file_name=str(image_index)+".png")

    # Plotting the embedded data:
    if plot_projected_pointsAndImages_again:
        if dataset == "ATT_glasses":
            scale = 1
            dataset_notReshaped = np.zeros((n_samples, image_height*scale, image_width*scale))
            for image_index in range(n_samples):
                image = data_notNormalized[:, image_index]
                image_not_reshaped = image.reshape((image_height, image_width))
                image_not_reshaped_scaled = scipy.misc.imresize(arr=image_not_reshaped, size=scale*100)
                dataset_notReshaped[image_index, :, :] = image_not_reshaped_scaled
            fig, ax = plt.subplots(figsize=(10, 10))
            # ---- only take two dimensions to plot:
            # plot_components(X_projected=data_transformed, which_dimensions_to_plot=which_dimensions_to_plot_inpointsAndImagesPlot,
            #                     images=255-dataset_notReshaped, ax=ax, image_scale=0.25, markersize=10, thumb_frac=0.07, cmap='gray_r')
            # class_legends = ["no glasses", "glasses"]
            class_legends = None
            plot_components_by_colors(X_projected=data_transformed, y_projected=Y_train, which_dimensions_to_plot=which_dimensions_to_plot_inpointsAndImagesPlot, ax=None, markersize=8, class_legends=class_legends, colors=["b", "r"], markers=["s", "o"])
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_componentsAndImages_2(X_projected=data_transformed, Y_projected=Y_train, which_dimensions_to_plot=which_dimensions_to_plot_inpointsAndImagesPlot,
                                images=255-dataset_notReshaped, ax=ax, image_scale=0.25, markersize=10, thumb_frac=0.01, cmap='gray_r')
        elif dataset == "Binary_XOR" or dataset == "Concentric_rings":
            # class_legends = ["class 1", "class 2"]
            class_legends = None
            plot_components_by_colors_withTest(X_train=data_transformed, X_test=data_outOfSample_transformed, y_train=Y_train, y_test=Y_test, which_dimensions_to_plot=[0,1], class_legends=class_legends, markersize=8, colors=["b", "r"], markers=["s", "o"])
        elif dataset == "MNIST":
            class_legends = None
            plot_components_by_colors_withTest(X_train=data_transformed, X_test=data_outOfSample_transformed, y_train=Y_train, y_test=Y_test, which_dimensions_to_plot=[0,1], class_legends=class_legends, markersize=8, colors=None, markers=None)

    # classification (1 nearest neighbor):
    if dataset == 'MNIST':
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        max_n_components = 30
        error = np.zeros((max_n_components,))
        for n_component in range(1,max_n_components+1):
            data_transformed_truncated = data_transformed[:n_component, :]
            data_outOfSample_transformed_truncated = data_outOfSample_transformed[:n_component, :]
            neigh = KNeighborsClassifier(n_neighbors=1)
            neigh.fit(X=data_transformed_truncated.T, y=Y_train.ravel())
            predicted_Y_test = neigh.predict(X=data_outOfSample_transformed_truncated.T)
            error[n_component-1] = sum(predicted_Y_test != Y_test) / len(Y_test)
            print("#components: "+str(n_component)+", error: "+str(error[n_component-1]))
        save_variable(error, 'error', path_to_save="./output/"+dataset+"/"+manifold_learning_method+"/r_1="+str(r_1)+", r_2="+str(r_2)+"/")
        save_np_array_to_txt(variable=error, name_of_variable="error", path_to_save="./output/"+dataset+"/"+manifold_learning_method+"/r_1="+str(r_1)+", r_2="+str(r_2)+"/")



def convert_mat_to_csv(path_mat, path_to_save):
    # https://gist.github.com/Nixonite/bc2f69b0c4430211bcad
    data = scipy.io.loadmat(path_mat)
    for i in data:
        if '__' not in i and 'readme' not in i:
            np.savetxt((path_to_save + i + ".csv"), data[i], delimiter=',')

def read_csv_file(path):
    # https://stackoverflow.com/questions/46614526/how-to-import-a-csv-file-into-a-data-array
    with open(path, newline='') as csvfile:
        data = list(csv.reader(csvfile))
    # convert to numpy array:
    data = np.asarray(data)
    return data

def load_image(address_image):
    # http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/
    img = Image.open(address_image).convert('L')
    img_arr = np.array(img)
    return img_arr

def save_image(image_array, path_without_file_name, file_name):
    if not os.path.exists(path_without_file_name):
        os.makedirs(path_without_file_name)
    # http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/
    img = Image.fromarray(image_array)
    img = img.convert("L")
    img.save(path_without_file_name + file_name)

def show_image(img):
    plt.imshow(img)
    plt.gray()
    plt.show()

def plot_components_by_colors(X_projected, y_projected, which_dimensions_to_plot, class_legends=None, colors=None, markers=None, ax=None, markersize=10):
    # X_projected: rows are features and columns are samples
    # which_dimensions_to_plot: a list of two integers, index starting from 0
    X_projected = np.vstack((X_projected[which_dimensions_to_plot[0], :], X_projected[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_projected = X_projected.T
    ax = ax or plt.gca()
    labels_of_classes = list(set(y_projected.ravel()))
    labels_of_classes.sort()  # --> sort ascending
    n_classes = len(labels_of_classes)
    if class_legends is None:
        class_legends_ = [""] * n_classes
    else:
        class_legends_ = class_legends
    if colors is None:
        colors_ = get_spaced_colors(n=n_classes)
    if markers is None:
        markers = ["o"] * n_classes
    for class_index in range(n_classes):
        class_label = labels_of_classes[class_index].astype(int)
        mask = (y_projected == class_label)
        mask = (mask == 1).ravel().tolist()
        X_projected_class = X_projected[mask, :]
        if colors is None:
            color = [colors_[class_index][0] / 255, colors_[class_index][1] / 255, colors_[class_index][2] / 255]
        else:
            color = colors[class_index]
        marker = markers[class_index]
        ax.plot(X_projected_class[:, 0], X_projected_class[:, 1], marker=marker, color=color, markersize=markersize, label=class_legends_[class_index], linestyle="None")
    plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    if class_legends is not None:
        ax.legend()
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_components_by_colors_withTest(X_train, X_test, y_train, y_test, which_dimensions_to_plot, class_legends=None, colors=None, markers=None, ax=None, markersize=10):
    # X_projected: rows are features and columns are samples
    # which_dimensions_to_plot: a list of two integers, index starting from 0
    X_train = np.vstack((X_train[which_dimensions_to_plot[0], :], X_train[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_test = np.vstack((X_test[which_dimensions_to_plot[0], :], X_test[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_train = X_train.T
    X_test = X_test.T
    ax = ax or plt.gca()
    y_train = np.asarray(y_train)
    labels_of_classes = list(set(y_train.ravel()))
    labels_of_classes.sort()  # --> sort ascending
    n_classes = len(labels_of_classes)
    if class_legends is None:
        class_legends_ = [""] * n_classes
    else:
        class_legends_ = class_legends
    if colors is None:
        colors_ = get_spaced_colors(n=n_classes)
    if markers is None:
        markers = ["o"] * n_classes
    # plot training data:
    for class_index in range(n_classes):
        class_label = labels_of_classes[class_index].astype(int)
        mask = (y_train == class_label)
        mask = (mask == 1).ravel().tolist()
        X_train_class = X_train[mask, :]
        if colors is None:
            color = [colors_[class_index][0] / 255, colors_[class_index][1] / 255, colors_[class_index][2] / 255]
        else:
            color = colors[class_index]
        marker = markers[class_index]
        ax.plot(X_train_class[:, 0], X_train_class[:, 1], marker=marker, color=color, markersize=markersize, label="training, "+class_legends_[class_index], linestyle="None")
    # plot test data:
    for class_index in range(n_classes):
        class_label = labels_of_classes[class_index].astype(int)
        mask = (y_test == class_label)
        mask = (mask == 1).ravel().tolist()
        X_train_class = X_test[mask, :]
        if colors is None:
            color = [colors_[class_index][0] / 255, colors_[class_index][1] / 255, colors_[class_index][2] / 255]
        else:
            color = colors[class_index]
        marker = markers[class_index]
        ax.plot(X_train_class[:, 0], X_train_class[:, 1], marker=marker, color=color, markersize=markersize, label="test, "+class_legends_[class_index], linestyle="None", fillstyle='none')  #--> fillstyle = ('full', 'left', 'right', 'bottom', 'top', 'none')
    # plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    # plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    if class_legends is not None:
        ax.legend()
    plt.xticks([])
    plt.yticks([])
    plt.show()

def get_spaced_colors(n):
    # https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

def plot_components(X_projected, which_dimensions_to_plot, images=None, ax=None, image_scale=1, markersize=10, thumb_frac=0.05, cmap='gray'):
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
    # X_projected: rows are features and columns are samples
    # which_dimensions_to_plot: a list of two integers, index starting from 0
    X_projected = np.vstack((X_projected[which_dimensions_to_plot[0], :], X_projected[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_projected = X_projected.T
    ax = ax or plt.gca()
    ax.plot(X_projected[:, 0], X_projected[:, 1], '.k', markersize=markersize)
    images = resize(images, (images.shape[0], images.shape[1]*image_scale, images.shape[2]*image_scale), order=5, preserve_range=True, mode="constant")
    if images is not None:
        min_dist_2 = (thumb_frac * max(X_projected.max(0) - X_projected.min(0))) ** 2
        shown_images = np.array([2 * X_projected.max(0)])
        for i in range(X_projected.shape[0]):
            dist = np.sum((X_projected[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, X_projected[i]])
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), X_projected[i])
            ax.add_artist(imagebox)
        # # plot the first (original) image once more to be on top of other images:
        # # change color of frame (I googled: python OffsetImage highlight frame): https://stackoverflow.com/questions/40342379/show-images-in-a-plot-using-matplotlib-with-a-coloured-frame
        # imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[0], cmap=cmap), X_projected[0], bboxprops =dict(edgecolor='red'))
        # ax.add_artist(imagebox)
    plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    plt.show()

def plot_componentsAndImages_with_test(X_projected, X_test_projected, which_dimensions_to_plot, images=None, images_test=None, ax=None, image_scale=1, markersize=10, thumb_frac=0.05, cmap='gray'):
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
    # X_projected: rows are features and columns are samples
    # which_dimensions_to_plot: a list of two integers, index starting from 0
    n_test_samples = X_test_projected.shape[1]
    X_projected = np.vstack((X_projected[which_dimensions_to_plot[0], :], X_projected[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_test_projected = np.vstack((X_test_projected[which_dimensions_to_plot[0], :], X_test_projected[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_projected = X_projected.T
    X_test_projected = X_test_projected.T
    ax = ax or plt.gca()
    ax.plot(X_projected[:, 0], X_projected[:, 1], '.k', markersize=markersize)
    ax.plot(X_test_projected[:, 0], X_test_projected[:, 1], '.r', markersize=markersize)
    images = resize(images, (images.shape[0], images.shape[1]*image_scale, images.shape[2]*image_scale), order=5, preserve_range=True, mode="constant")
    images_test = resize(images_test, (images_test.shape[0], images_test.shape[1]*image_scale, images_test.shape[2]*image_scale), order=5, preserve_range=True, mode="constant")
    if images is not None:
        min_dist_2 = (thumb_frac * max(X_projected.max(0) - X_projected.min(0))) ** 2
        shown_images = np.array([2 * X_projected.max(0)])
        for i in range(X_projected.shape[0]):
            # test image:
            if i < n_test_samples:
                dist = np.sum((X_test_projected[i] - shown_images) ** 2, 1)
                if np.min(dist) < min_dist_2:
                    # don't show points that are too close
                    continue
                shown_images = np.vstack([shown_images, X_test_projected[i]])
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images_test[i], cmap=cmap), X_test_projected[i], bboxprops =dict(edgecolor='red', lw=3))
                ax.add_artist(imagebox)
            # training image:
            dist = np.sum((X_projected[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, X_projected[i]])
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), X_projected[i])
            ax.add_artist(imagebox)
        # # plot the first (original) image once more to be on top of other images:
        # # change color of frame (I googled: python OffsetImage highlight frame):
        # https://stackoverflow.com/questions/40342379/show-images-in-a-plot-using-matplotlib-with-a-coloured-frame
        # https://matplotlib.org/users/annotations_guide.html
        # imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[0], cmap=cmap), X_projected[0], bboxprops =dict(edgecolor='red'))
        # ax.add_artist(imagebox)
    plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    plt.show()

def plot_componentsAndImages_2(X_projected, Y_projected, which_dimensions_to_plot, images=None, ax=None, image_scale=1, markersize=10, thumb_frac=0.05, cmap='gray'):
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
    # X_projected: rows are features and columns are samples
    # which_dimensions_to_plot: a list of two integers, index starting from 0
    X_projected = np.vstack((X_projected[which_dimensions_to_plot[0], :], X_projected[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_projected = X_projected.T
    ax = ax or plt.gca()
    Y_projected = Y_projected.ravel()
    ax.plot(X_projected[Y_projected == 0, 0], X_projected[Y_projected == 0, 1], '.k', markersize=markersize)
    ax.plot(X_projected[Y_projected == 1, 0], X_projected[Y_projected == 1, 1], '.r', markersize=markersize)
    images = resize(images, (images.shape[0], images.shape[1]*image_scale, images.shape[2]*image_scale), order=5, preserve_range=True, mode="constant")
    if images is not None:
        min_dist_2 = (thumb_frac * max(X_projected.max(0) - X_projected.min(0))) ** 2
        shown_images = np.array([2 * X_projected.max(0)])
        for i in range(X_projected.shape[0]):
            dist = np.sum((X_projected[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, X_projected[i]])
            if Y_projected[i] == 0:
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), X_projected[i])
            else:
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), X_projected[i], bboxprops =dict(edgecolor='red', lw=3))
            ax.add_artist(imagebox)
        # # plot the first (original) image once more to be on top of other images:
        # # change color of frame (I googled: python OffsetImage highlight frame): https://stackoverflow.com/questions/40342379/show-images-in-a-plot-using-matplotlib-with-a-coloured-frame
        # imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[0], cmap=cmap), X_projected[0], bboxprops =dict(edgecolor='red'))
        # ax.add_artist(imagebox)
    plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    plt.show()

def save_variable(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.pckl'
    f = open(file_address, 'wb')
    pickle.dump(variable, f)
    f.close()

def save_np_array_to_txt(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.txt'
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    with open(file_address, 'w') as f:
        f.write(np.array2string(variable, separator=', '))

def load_variable(name_of_variable, path='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    file_address = path + name_of_variable + '.pckl'
    f = open(file_address, 'rb')
    variable = pickle.load(f)
    f.close()
    return variable

def read_facebook_dataset(path_dataset, sep=";", header='infer'):
    # return: X, Y --> rows: features, columns = samples
    data = pd.read_csv(path_dataset, sep=sep, header=header)
    # print(list(data.columns.values))
    names_of_input_features = ["Category", "Page total likes", "Type", "Post Month", "Post Hour", "Post Weekday", "Paid"]
    names_of_output_features = ["Lifetime Post Total Reach", "Lifetime Post Total Impressions", "Lifetime Engaged Users", "Lifetime Post Consumers", "Lifetime Post Consumptions", "Lifetime Post Impressions by people who have liked your Page", "Lifetime Post reach by people who like your Page", "Lifetime People who have liked your Page and engaged with your post", "comment", "like", "share", "Total Interactions"]
    X = np.zeros((data.shape[0], len(names_of_input_features)))
    for feature_index, feature_name in enumerate(names_of_input_features):
        try:
            X[:, feature_index] = data.loc[:, feature_name].values
        except:  # feature if categorical
            feature_vector = data.loc[:, feature_name]
            le = preprocessing.LabelEncoder()
            le.fit(feature_vector)
            X[:, feature_index] = le.transform(feature_vector)
    X = X.T
    Y = np.zeros((data.shape[0], len(names_of_output_features)))
    for feature_index, feature_name in enumerate(names_of_output_features):
        Y[:, feature_index] = data.loc[:, feature_name].values
    Y = Y.T
    # Five samples have some nan values, such as: the "Paid" feature of last sample (X[-1,-1]) is missing and thus nan. We remove it:
    indices_of_samples_not_having_missing_values = np.logical_and(~np.isnan(X).any(axis=0), ~np.isnan(Y).any(axis=0))  # https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-91.php
    X = X[:, indices_of_samples_not_having_missing_values]
    Y = Y[:, indices_of_samples_not_having_missing_values]
    return X, Y

def read_regression_dataset_withOneOutput(path_dataset, column_of_labels, sep=";", header='infer'):
    # return: X, Y --> rows: features, columns = samples
    data = pd.read_csv(path_dataset, sep=sep, header=header)
    X = np.zeros(data.shape)
    for feature_index in range(data.shape[1]):
        try:
            X[:, feature_index] = data.iloc[:, feature_index].values
        except:  # feature if categorical
            feature_vector = data.iloc[:, feature_index]
            le = preprocessing.LabelEncoder()
            le.fit(feature_vector)
            X[:, feature_index] = le.transform(feature_vector)
    Y = X[:, column_of_labels]
    X = np.delete(X, column_of_labels, 1)  # delete the column of labels from X
    X = X.T
    Y = Y.T
    return X, Y

if __name__ == '__main__':
    main()