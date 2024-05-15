import pandas as pd
from data import DataBase
import numpy as np
import os


class Network:
    def __init__(self, extra_layers, read_data_from_files=False) -> None:
        """
        Creates network with randmo weights and biases.
        Network will have structure 784 -> extra_layers -> 10
        Parameters:
        list_network_nodes: list of extra layers with neurons
        """
        self.data_base = DataBase(read_data_from_files)
        self.train_data, self.test_data = self.data_base.get_train_test_set()

        self.sizes = [784] + extra_layers + [10]
        self.num_layers = len(self.sizes)
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.label_matirx = np.eye(10)

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def feedforward(self, x):
        """
        Feeds data through network
        Returns:
        vector of size last_layer with activations of each neuron in this layer
        """
        a = np.expand_dims(x, axis=-1)
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.sigmoid(z)
        return a

    def SGD(self, epochs, mini_batch_size, eta, test_data=None):
        """
        Stochastic gradient descent to train the network.
        Calculates optimal weights and biases
        Parameters:
        epochs: number of iterations of stochastic gradient descent
        mini_batch_size: size of batches of data 
        eta: learnig rate
        test_data: if provided will evaluate after each epoch
        """
        training_data = self.train_data
        if test_data:
            n_test = len(test_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            for i in range(0, len(training_data), mini_batch_size):
                self.update_mini_batch(training_data[i:i+mini_batch_size], eta)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        Updates network based on avrage gradient of mini batch * learning rate
        """
        gradient_weights = [np.zeros(w.shape) for w in self.weights]
        gradient_biases = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            d_w, d_b = self.backprop(x, y)
            gradient_weights = [gw + dw for gw,
                                dw in zip(gradient_weights, d_w)]
            gradient_biases = [gb + db for gb, db in zip(gradient_biases, d_b)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, gradient_weights)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, gradient_biases)]

    def backprop(self, x, y):
        """
        Backpropagate in order to calculate gradient for one example
        Parameters:
        x: input
        y: label
        Returns:
        gradinent of weights, gradient of biases
        """
        a = np.expand_dims(x, axis=-1)
        a_lst = [a]
        z_lst = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.sigmoid(z)
            z_lst.append(z)
            a_lst.append(a)
        y_vec = self.label_matirx[:, y][:, np.newaxis]
        delta = [(a_lst[-1] - y_vec)
                 * self.sigmoid_prime(z_lst[-1])]
        d_weights = [np.dot(delta[0], a_lst[-2].T)]
        for l in range(2, self.num_layers):
            z = z_lst[-l]
            sp = self.sigmoid_prime(z)
            new_delta = np.dot(self.weights[-l+1].transpose(), delta[0]) * sp
            delta.insert(0, new_delta)
            d_weights.insert(0, np.dot(new_delta, a_lst[-l-1].T))

        return d_weights, delta

    def evaluate(self, test_data=None):
        """
        Evaluates network based on test data
        Returns:
        correct guesses/test data
        """
        if test_data == None:
            test_data = self.test_data
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)/len(test_data)

    def predict(self, x):
        """returns digit and certainity"""
        ff_result = self.feedforward(x)
        return np.argmax(ff_result), np.max(ff_result)

    def save_network(self, dir='network'):
        """
        Saves network to local directory if not provided uses ./network
        """
        dir = './'+dir
        if not os.path.exists(dir):
            os.makedirs(dir)

        for i, weight_array in enumerate(self.weights):
            np.save(os.path.join(dir, f'weight_{i}.npy'), weight_array)

        for i, bias_array in enumerate(self.biases):
            np.save(os.path.join(dir, f'bias_{i}.npy'), bias_array)

    def load_network(self, dir='network'):
        """
        Loads network from local directory if not provided uses ./network
        """
        dir = './'+dir
        if not os.path.exists(dir):
            return
        weight_files = [file for file in os.listdir(
            dir) if file.startswith('weight')]
        bias_files = [file for file in os.listdir(
            dir) if file.startswith('bias')]

        weight_arrays = [np.load(os.path.join(dir, file))
                         for file in weight_files]
        bias_arrays = [np.load(os.path.join(dir, file)) for file in bias_files]

        self.weights = weight_arrays
        self.biases = bias_arrays


if __name__ == "__main__":
    network = Network([30], False)
    network.SGD(30, 10, 3)
    network.save_network()
