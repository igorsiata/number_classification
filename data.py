import pandas as pd
from sklearn.datasets import fetch_openml
import numpy as np
import os


class DataBase:
    def __init__(self, read_from_files=False) -> None:

        if read_from_files:
            self.X_train, self.X_test, self.y_train, self.y_test = self.read_data()
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = self.load_data()

    def load_data(self):
        """
        Loads data from online database -> binarizes the images -> splits it to test and train set
        Returns: training and testing sets with labels
        """
        mnist = fetch_openml('mnist_784', parser='auto',
                             version=1, as_frame=False)
        X, y = mnist["data"], mnist["target"]
        X.shape
        X = X/255.
        X = np.where(X > 0.5, 1, 0)
        y = y.astype(np.uint8)
        X_train, X_test, y_train, y_test = X[:
                                             60000], X[60000:], y[:60000], y[60000:]
        return X_train, X_test, y_train, y_test

    def save_data(self, dir='data') -> None:
        """
        Saves data to local directory, if not specified uses directory ./data
        """
        dir = './'+dir
        if not os.path.exists(dir):
            os.makedirs(dir)
        np.save(os.path.join(dir, 'X_train'), self.X_train)
        np.save(os.path.join(dir, 'X_test'), self.X_test)
        np.save(os.path.join(dir, 'y_train'), self.y_train)
        np.save(os.path.join(dir, 'y_test'), self.y_test)

    def read_data(self, dir='data') -> None:
        """
        Reads data from local directory, if not specified uses directory ./data
        Writes the data to class members 
        """
        dir = './'+dir
        if not os.path.exists(dir):
            return
        X_train = np.load(os.path.join(dir, 'X_train.npy'))
        X_test = np.load(os.path.join(dir, 'X_test.npy'))
        y_train = np.load(os.path.join(dir, 'y_train.npy'))
        y_test = np.load(os.path.join(dir, 'y_test.npy'))
        return X_train, X_test, y_train, y_test

    def get_train_test_set(self):
        """
        Merges data with labels to create train and test set that are:
        list(tuple(data, label))
        """
        train_set = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        test_set = [(x, y) for x, y in zip(self.X_test, self.y_test)]
        return train_set, test_set


if __name__ == "__main__":
    data = DataBase()
    data.save_data()
