import numpy as np
from numpy import linalg as LA
from sklearn.metrics.pairwise import pairwise_kernels
from my_generalized_eigen_problem import My_generalized_eigen_problem

class My_RDA:

    def __init__(self, r_1=0, r_2=0, n_components=None, kernel_on_labels=None):
        self.n_components = n_components
        self.n_samples_training = None
        self.dimensionality = None
        self.P = None
        self.Sw = None
        self.K_y = None
        self.R1 = None
        self.R2 = None
        self.n_classes = None #--> used for classification task
        self.U = None
        self.r_1 = r_1
        self.r_2 = r_2
        if kernel_on_labels != None:
            self.kernel_on_labels = kernel_on_labels
        else:
            self.kernel_on_labels = "delta_kernel"

    def calculate_P(self, Y):
        # Y: rows are dimensions of labels (usually 1-dimensional) and columns are samples
        if self.r_1 != 0:
            if self.kernel_on_labels == "delta_kernel":
                self.K_y = self.delta_kernel(Y=Y)
            else:
                self.K_y = pairwise_kernels(X=Y.T, Y=Y.T, metric=self.kernel_on_labels)
        else:
            self.K_y = np.zeros((self.n_samples_training, self.n_samples_training))
        P = (self.r_1 * self.K_y) + ((1 - self.r_1) * np.eye(self.n_samples_training))
        return P

    def calculate_R1(self, X, Y):
        # X: rows are features, columns are samples
        # Y: rows are dimensions of labels (usually 1-dimensional) and columns are samples
        self.P = self.calculate_P(Y=Y)
        n = X.shape[1]
        H = np.eye(n) - ((1 / n) * np.ones((n, n)))
        R1 = X.dot(H).dot(self.P).dot(H).dot(X.T)
        return R1

    def calculate_Sw(self, X, Y):
        # Y: rows are dimensions of labels (must be 1-dimensional here) and columns are samples
        y = np.asarray(Y)
        y = y.reshape((1, -1))
        labels_of_classes = list(set(y.ravel()))
        self.n_classes = len(labels_of_classes)
        X_separated_classes = self._separate_samples_of_classes(X=X, y=Y)
        Sw = np.zeros((self.dimensionality, self.dimensionality))
        for class_index in range(self.n_classes):
            print("Calculating Sw: class " + str(class_index))
            X_class = X_separated_classes[class_index]
            n_samples_of_class = X_class.shape[1]
            mean_of_class = X_class.mean(axis=1)
            mean_of_class = mean_of_class.reshape((-1, 1))
            X_class_centered = X_class - mean_of_class
            for sample_index in range(n_samples_of_class):
                print("Calculating Sw: sample " + str(sample_index))
                temp = X_class_centered[:, sample_index]
                Sw = Sw + temp.dot(temp.T)
        return Sw

    def calculate_R2(self, X, Y):
        # X: rows are features, columns are samples
        # Y: rows are dimensions of labels (usually 1-dimensional) and columns are samples
        if self.r_2 != 0:
            self.Sw = self.calculate_Sw(X=X, Y=Y)
        else:
            self.Sw = np.zeros((self.dimensionality, self.dimensionality))
        R2 = (self.r_2 * self.Sw) + ((1 - self.r_2) * np.eye(self.dimensionality))
        return R2

    def delta_kernel(self, Y):
        Y = Y.ravel()
        n_samples = len(Y)
        delta_kernel = np.zeros((n_samples, n_samples))
        for sample_index_1 in range(n_samples):
            for sample_index_2 in range(n_samples):
                if Y[sample_index_1] == Y[sample_index_2]:
                    delta_kernel[sample_index_1, sample_index_2] = 1
                else:
                    delta_kernel[sample_index_1, sample_index_2] = 0
        return delta_kernel

    def fit_transform(self, X, Y):
        # X: rows are features and columns are samples
        # Y: rows are dimensions of labels (usually 1-dimensional) and columns are samples
        self.fit(X, Y)
        X_transformed = self.transform(X, Y)
        return X_transformed

    def fit(self, X, Y):
        # X: rows are features and columns are samples
        # Y: rows are dimensions of labels (usually 1-dimensional) and columns are samples
        self.n_samples_training = X.shape[1]
        self.dimensionality = X.shape[0]
        print("Calculating R1...")
        self.R1 = self.calculate_R1(X=X, Y=Y)
        print("R1 calculated. Calculating R2...")
        self.R2 = self.calculate_R2(X=X, Y=Y)
        print("R2 calculated. Calculating eigenvectors...")
        my_generalized_eigen_problem = My_generalized_eigen_problem(A=self.R1, B=self.R2)
        eig_vec, eig_val = my_generalized_eigen_problem.solve()
        # eig_vec, eig_val = my_generalized_eigen_problem.solve_dirty()
        print("Eigenvectors calculated.")
        # idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        # eig_val = eig_val[idx]
        # eig_vec = eig_vec[:, idx]
        if self.n_components is not None:
            self.U = eig_vec[:, :self.n_components]
        else:
            self.U = eig_vec

    def transform(self, X, Y=None):
        # X: rows are features and columns are samples
        X_transformed = (self.U.T).dot(X)
        return X_transformed

    def get_projection_directions(self):
        return self.U

    def reconstruct(self, X, scaler=None, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        if using_howMany_projection_directions != None:
            U = self.U[:, 0:using_howMany_projection_directions]
        else:
            U = self.U
        X_transformed = (U.T).dot(X)
        X_reconstructed = U.dot(X_transformed)
        if scaler is not None:
            X_reconstructed = scaler.inverse_transform(X=X_reconstructed.T)
            X_reconstructed = X_reconstructed.T
        return X_reconstructed

    def center_the_matrix(self, the_matrix, mode="double_center"):
        n_rows = the_matrix.shape[0]
        n_cols = the_matrix.shape[1]
        vector_one_left = np.ones((n_rows,1))
        vector_one_right = np.ones((n_cols, 1))
        H_left = np.eye(n_rows) - ((1/n_rows) * vector_one_left.dot(vector_one_left.T))
        H_right = np.eye(n_cols) - ((1 / n_cols) * vector_one_right.dot(vector_one_right.T))
        if mode == "double_center":
            the_matrix = H_left.dot(the_matrix).dot(H_right)
        elif mode == "remove_mean_of_rows_from_rows":
            the_matrix = H_left.dot(the_matrix)
        elif mode == "remove_mean_of_columns_from_columns":
            the_matrix = the_matrix.dot(H_right)
        return the_matrix

    def _separate_samples_of_classes(self, X, y):
        # X --> rows: features, columns: samples
        # X_separated_classes --> rows: features, columns: samples
        y = np.asarray(y)
        y = y.reshape((1, -1))
        labels_of_classes = list(set(y.ravel()))
        labels_of_classes.sort()  #--> sort ascending
        n_classes = len(labels_of_classes)
        X_separated_classes = [None] * n_classes
        for class_index in range(n_classes):
            label = labels_of_classes[class_index]
            mask = (y == label)
            mask = (mask==1).ravel().tolist()
            X_class = X[:, mask]
            X_separated_classes[class_index] = X_class
        return X_separated_classes