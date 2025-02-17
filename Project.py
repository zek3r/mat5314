import sys
from pathlib import Path
import os as os
import numpy as np
import scipy.io as sio
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

# Functions-------------------------------------------------------------------------------------------------------------


def load_mnist(name):
    """
    Function loads MNIST data set. Assumes that the data file is contained in the working directory
    :param name: string, name of file including extension
    :return: data: MNIST data set.
    """
    working_dir = Path.cwd()
    get_file = working_dir / name
    data = sio.loadmat(get_file)

    return data


def init_mat(element_sd, p_connection, num_features, num_outputs):
    """
    Initializes connectivity matrix W. element_sd=1; p_connection=1; num_features=d; num_outputs=k;
    :param element_sd: Standard deviation of each element
    :param p_connection: probability that a single element is non-zero
    :param num_features: (d) number of features (size of input layer)
    :param num_outputs: (o) number of outputs (size of output layer
    :return: W: d x o connectivity matrix
    """

    W = np.random.binomial(1, p_connection, (num_outputs, num_features))
    X = np.random.normal(0, element_sd, (num_outputs, num_features))
    W = (W * X)/np.sqrt(num_features)

    return W


def init_bias(num_outputs, element_mean=None, element_sd=None):
    """
    Initializes bias vector b. num_outputs=k; element_mean=0.001; element_sd=0
    :param num_outputs: (o) size of bias vector
    :param element_mean: mean of elements. If None mean is zero
    :param element_sd: sd of elements. If None vector is constant. If both params are None then zero vector returned
    :return: bias vector
    """
    if element_mean and element_sd:
        b = np.random.normal(element_mean, element_sd, num_outputs)
    elif element_mean:
        b = np.ones(num_outputs)*element_mean
    elif element_sd:
        b = np.random.normal(0, element_sd, num_outputs)
    else:
        b = np.zeros(num_outputs)

    return b


def softmax(z, axis=0):
    """
    Generates softmax of input vector z
    :param z: input array
    :param axis: which axis to calculate softmax over (0=col, 1=row) if z is 2d
    :return: softmax of z
    """

    if len(z.shape) == 1:
        c = np.max(z)
        s = np.exp(z - c)/np.sum(np.exp(z - c))
    elif len(z.shape) == 2:
        s = np.zeros(z.shape)

        if axis == 0:
            for i in range(z.shape[1]):
                c = np.max(z[:, i])
                s[:, i] = np.exp(z[:, i] - c)/np.sum(np.exp(z[:, i] - c))

                if np.all(np.isnan(s[:, i])):
                    s[:, i] = np.zeros(s[:, i].shape)
                    s[np.argmax(z[:, i]), i] = 1.

        if axis == 1:
            for i in range(z.shape[0]):
                c = np.max(z[i, :])
                s[i, :] = np.exp(z[i, :] - c)/np.sum(np.exp(z[i, :] - c))
    else:
        print('Invalid input dimension for softmax')
        sys.exit()

    return s


def lnn_update(W, b, x, y):
    """
    Calculates parameter updates for linear neural network for input data x and target values y where k is num outputs,
    d is num features and m is num samples using gradient descent of log likelihood
    :param W: connectivity matrix k x d
    :param b: bias vector (must be k x 1)
    :param x: input data matrix. This should be d x m
    :param y: input target value matrix. This should be k x m
    :return: dW: change in values of weight matrix according to gradient descent of negative of log likelihood
    :return: db: change in values of bias vector according to gradient descent of negative of log likelihood
    """

    m = y.shape[1]  # num_samples

    b = np.tile(b, (m, 1)).transpose()
    z = W @ x + b

    #s = sp.special.softmax(z, axis=0)
    s = softmax(z, axis=0)
    y_s = y - s
    db = np.sum(y_s, axis=1)
    dW = y_s @ x.transpose()

    return dW, db


def lnn_predict(W, b, x):
    """
    Predicts values of linear neural network for test data x where k is num outputs,
    d is num features and m is num samples
    :param W: connectivity matrix
    :param b: bias vector. Must be k x 1
    :param x: input data matrix. This should be d x m
    :return: p: probability that the jth (in 1,...,m) data point falls in the ith (in 1,...,k) class. This will be k x m
    """

    m = x.shape[1]
    b = np.tile(b, (m, 1)).transpose()

    #p = sp.special.softmax(W @ x + b, axis=0)
    p = softmax(W @ x + b, axis=0)

    return p


class linear_nn:

    def __init__(self, W, b):
        self.W = W
        self.b = b

    def params(self):
        """
        Returns parameters
        :return: W
        :return: b
        """
        return self.W, self.b

    def train(self, x, y, k, eta, batch_size, num_epochs):
        """
        Takes in n samples of training data with d features and trains network with k output classes. If batch_size
        doesn't divide n then the last batch contains the remaining samples.
        x=x_train; y=y_train; k=k; batch_size=n_batch; num_epochs=1
        :param x: training data. Should be d x n array
        :param y: target values. Should be 1 x n of integers array of indices of output classes
        :param k: number of output classes.
        :param eta: learning rate
        :param batch_size: integer value
        :param num_epochs: integer value
        :return: params: tuple containing trained W and b
        """

        n = len(y)
        y_mat = np.zeros((k, n), dtype=int)
        for i in range(n):
            y_mat[y[i], i] = 1                                                              # Create y_mat

        num_batches = int(n / batch_size) + 1

        for _ in range(num_epochs):                                                         # Loop through epochs
            index = np.arange(0, n, 1)

            for _ in range(num_batches):                                                    # Loop through batches
                if len(index) > batch_size:
                    samples = np.random.choice(index, size=batch_size, replace=False)
                    index = index[np.isin(index, samples, assume_unique=True, invert=True)]
                else:
                    samples = index

                y_batch = y_mat[:, samples]
                x_batch = x[:, samples]

                dW, db = lnn_update(self.W, self.b, x_batch, y_batch)
                self.W = self.W + eta*dW
                self.b = self.b + eta*db

        params = (self.W, self.b)

        return params

    def test(self, x):
        """
        Evaluates neural network on test data with d features and n samples
        :param x: test data. Should be d x n
        :return: p, class probabilities. k x n array
        :return: y_hat, predictions. k x n array
        """

        p = lnn_predict(self.W, self.b, x)
        y_hat = np.argmax(p, axis=0)

        return y_hat, p


def lnn_likelihood(W, b, k, x, y):
    """
    Calculates negative log likelihood for params W, b and data x,y based on 1-layer linear neural network architecture
    :param W: array, connectivity matrix
    :param b: vector, bias vector
    :param k: integer, number of output classes
    :param x: array, data samples
    :param y: vector, target values
    :return: ll, negative log likelihood
    """
    m = len(y)
    y_mat = np.zeros((k, m), dtype=int)
    for i in range(m):
        y_mat[y[i], i] = 1               # Create y_mat

    b = np.tile(b, (m, 1)).transpose()
    z = W @ x + b

    s = softmax(z, axis=0)

    ll = 0
    for i in range(m):
        ll = ll + -np.log((y_mat[:, i] @ s[:, i])/np.sum(s[:, i]))

    return ll


# Main Script-----------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    save_plots = True                                      # Choose whether or not to save plots
    test_gradient = True                                   # Choose whether or not to test gradient
    plot_in = os.getcwd() + '/plots/Project'

    data = load_mnist('mnist_all.mat')                      # Load

    # Setup-------------------------------------------------------------------------------------------------------------
    to_plot = []
    for key, value in data.items():                         # Rescale
        if key[0] == 't':
            data[key] = data[key]/255.0
            if key[1] == 'r':
                to_plot.append(value[0].reshape(28, 28))

    fig = plt.figure(1)                                     # Plot numbers
    fig.suptitle('Example Data Points')
    gs0 = gs.GridSpec(2, 5)
    index = [(x, y) for x in range(2) for y in range(5)]

    for i, ix in enumerate(index):
        ax = fig.add_subplot(gs0[ix])
        ax.imshow(to_plot[i], cmap='Greys')
        ax.axis('off')

    if save_plots:
        fig.savefig(plot_in + '/fig1', format='pdf')

    d = 28**2                                               # Initialize parameters
    k = 10
    learning_rate = 1
    W_sd = 1
    W_p = 1
    epochs = 10
    n_batch = 16
    W = init_mat(element_sd=W_sd, p_connection=W_p, num_features=d, num_outputs=k)
    b = init_bias(num_outputs=k, element_mean=0.001, element_sd=0)

    x_train = data['train0']                                # Initialize training variables
    for i in range(1, 10):
        x_train = np.concatenate((x_train, data['train' + str(i)]), axis=0)
    x_train = x_train.transpose()
    y_train = []
    for i in range(10):
        y_train.extend(np.ones(data['train' + str(i)].shape[0], dtype=int)*i)
    y_train = np.asarray(y_train)
    n = len(y_train)

    # Test Gradient Descent Algo.---------------------------------------------------------------------------------------
    if test_gradient:

        gt_m = 16                                           # Get data for gradient test
        gt_index = np.arange(n)
        gt_samples = np.random.choice(gt_index, size=gt_m, replace=False)
        y_gt = y_train[gt_samples]
        x_gt = x_train[:, gt_samples]
        k_gt = np.max(y_gt)
        y_mat_gt = np.zeros((k, gt_m), dtype=int)
        for i in range(gt_m):
            y_mat_gt[y_gt[i], i] = 1

        dW_gt, db_gt = lnn_update(W, b, x_gt, y_mat_gt)     # Analytically compute negative partial derivatives
        test_points_gt = np.where(dW_gt != 0)
        l_gt = len(test_points_gt[0])
        index_gt = np.arange(l_gt)
        samples_gt = np.random.choice(index_gt, 2, replace=False)
        ix_w1 = (test_points_gt[0][samples_gt[0]], test_points_gt[1][samples_gt[0]])
        ix_w2 = (test_points_gt[0][samples_gt[1]], test_points_gt[1][samples_gt[1]])
        w1_test = (ix_w1)
        w2_test = (ix_w2)
        b_test = 4

        dW1a = dW_gt[w1_test]
        dW2a = dW_gt[w2_test]
        db1a = db_gt[b_test]

        h_gt = np.array([0.9, 0.5, 0.4, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])
        lh = len(h_gt)                                      # Numerically compute neg. partial derivatives for 3 params
        diff_W1 = np.zeros(lh)
        diff_W2 = np.zeros(lh)
        diff_b = np.zeros(lh)
        for ix, h in enumerate(h_gt):
            W1 = np.copy(W)
            W2 = np.copy(W)
            b1 = np.copy(b)
            W1[w1_test] = W1[w1_test] + h
            W2[w2_test] = W2[w2_test] + h
            b1[b_test] = b1[b_test] + h

            f_a = lnn_likelihood(W, b, k, x_gt, y_gt)
            f_ah_w1 = lnn_likelihood(W1, b, k, x_gt, y_gt)
            f_ah_w2 = lnn_likelihood(W2, b, k, x_gt, y_gt)
            f_ah_b1 = lnn_likelihood(W, b1, k, x_gt, y_gt)

            dW1 = - (f_ah_w1 - f_a) / h
            dW2 = - (f_ah_w2 - f_a) / h
            db1 = - (f_ah_b1 - f_a) / h

            diff_W1[ix] = np.abs(dW1a - dW1)
            diff_W2[ix] = np.abs(dW2a - dW2)
            diff_b[ix] = np.abs(db1a - db1)

        fig2 = plt.figure(2)                                # Plot results of gradient test
        ax = fig2.add_subplot(111)
        ax.set_title('Difference Between Analytical and Numerical Partial Derivatives')
        ax.set_ylabel('Difference')
        ax.set_xlabel('-Log of Step (h) in Numerical Approx.')
        ax.plot(-np.log(h_gt), diff_W1, color='blue', ls='--')
        ax.plot(-np.log(h_gt), diff_W2, color='red', ls=':')
        ax.plot(-np.log(h_gt), diff_b, color='green', ls='-.')

        if save_plots:
            fig2.savefig(plot_in + '/fig2', format='pdf')

    # Train and Test Model----------------------------------------------------------------------------------------------
    x_test = data['test0']                                  # Initialize testing variables
    for i in range(1, 10):
        x_test = np.concatenate((x_test, data['test' + str(i)]), axis=0)
    x_test = x_test.transpose()
    y_test = []
    for i in range(10):
        y_test.extend(np.ones(data['test' + str(i)].shape[0], dtype=int)*i)
    y_test = np.asarray(y_test)

    lnn = linear_nn(W, b)                                   # Create neural network
    error = []                                              # Train network and plot error at each epoch
    for _ in range(epochs):
        learning_rate = learning_rate/2
        _ = lnn.train(x_train, y_train, k=k, eta=learning_rate, batch_size=n_batch, num_epochs=1)
        y_hat, _ = lnn.test(x_test)

        error.append(np.sum(np.invert(y_hat == y_test)))

    W, b = lnn.params()

    x_axis = np.arange(epochs)
    error = np.asarray(error)

    fig3 = plt.figure(3)
    ax = fig3.add_subplot(111)
    ax.set_title('Error on Test Set During Training')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Number of Errors')
    plt.plot(x_axis, error)

    if save_plots:
        fig3.savefig(plot_in + '/fig3', format='pdf')

    fig4 = plt.figure(4)
    ax = fig4.add_subplot(111)
    ax.set_title('Weights to o3')
    plot_mat = W[3, :].reshape(28, 28)
    ax.imshow(plot_mat, cmap='plasma')
    ax.axis('off')

    if save_plots:
        fig4.savefig(plot_in + '/fig4', format='pdf')

    # Keras-------------------------------------------------------------------------------------------------------------
    k_epochs = 10
    fnn = Sequential()                                      # Create neural network
    fnn.add(Dense(300, activation='tanh', input_dim=d))
    fnn.add(Dropout(0.5))
    fnn.add(Dense(k, activation='softmax'))

    sgd = SGD(lr=0.1, momentum=0.3, decay=0, nesterov=True)
    fnn.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    y_mat_train = np.zeros((k, len(y_train)), dtype=int)
    for i in range(len(y_train)):
        y_mat_train[y_train[i], i] = 1
    y_mat_test = np.zeros((k, len(y_test)), dtype=int)
    for i in range(len(y_test)):
        y_mat_test[y_test[i], i] = 1

    fnn.fit(x_train.transpose(), y_mat_train.transpose(), batch_size=32, epochs=k_epochs)
    res = fnn.evaluate(x_test.transpose(), y_mat_test.transpose(), batch_size=32)

    x_axis = np.arange(1, k_epochs+1)
    test_loss = res[0]
    test_acc = res[1]
    y_axis = np.ones(len(x_axis))
    y_loss = y_axis*test_loss
    y_acc = y_axis*test_acc

    gs1 = gs.GridSpec(1, 2)
    fig5 = plt.figure(5, (8, 4))
    fig5.suptitle('FNN Learning Curves')
    ax1 = fig5.add_subplot(gs1[0,0])
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.plot(x_axis, fnn.history.history['loss'], color='red')

    ax2 = fig5.add_subplot(gs1[0,1])
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.plot(x_axis, fnn.history.history['acc'], color='blue')

    gs1.tight_layout(fig5, rect=[0, 0.03, 1, 0.95])

    if save_plots:
        fig5.savefig(plot_in + '/fig5', format='pdf')

    fig6 = plt.figure(6, (8, 4), tight_layout=True)
    gs2 = gs.GridSpec(1,2)

    visualize1 = fnn.layers[0].get_weights()[0][:, 99].reshape(28, 28)
    ax1 = fig6.add_subplot(gs2[0,0])
    ax1.set_title('Weights to Hidden Unit 100')
    ax1.imshow(visualize1, cmap='plasma')
    ax1.axis('off')

    visualize2 = fnn.layers[0].get_weights()[0][:, 199].reshape(28, 28)
    ax2 = fig6.add_subplot(gs2[0,1])
    ax2.set_title('Weights to Hidden Unit 200')
    ax2.imshow(visualize2, cmap='plasma')
    ax2.axis('off')

    if save_plots:
        fig6.savefig(plot_in + '/fig6',  format='pdf')

print('LNN Accuracy is: ' + str(1 - error[len(error) - 1]/10000) + '\n')
print('FNN Accuracy is: ' + str(res[1]))
