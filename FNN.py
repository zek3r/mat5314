from pathlib import Path
import os as os
import numpy as np
import random as rn
import scipy.io as sio
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
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


# Main Script-----------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    save_plots = True                                      # Choose whether or not to save plots
    test_gradient = True                                   # Choose whether or not to test gradient

    #plot_in = os.getcwd() + '/plots'
    plot_in = '/Users/macbookair/desktop/MAT5314/tex_stuff_proj_full'
    data = load_mnist('mnist_all.mat')                      # Load

    # Set seeds and force single threading for reproducible results-----------------------------------------------------
    np.random.seed(5314)
    rn.seed(53140)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(5314)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    # Setup-------------------------------------------------------------------------------------------------------------
    to_plot = []
    for key, value in data.items():                         # Rescale
        if key[0] == 't':
            data[key] = data[key]/255.0
            if key[1] == 'r':
                to_plot.append(value[0].reshape(28, 28))

    x_train = data['train0']                                # Initialize training variables
    for i in range(1, 10):
        x_train = np.concatenate((x_train, data['train' + str(i)]), axis=0)
    x_train = x_train.transpose()
    y_train = []
    for i in range(10):
        y_train.extend(np.ones(data['train' + str(i)].shape[0], dtype=int)*i)
    y_train = np.asarray(y_train)
    n = len(y_train)
    x_test = data['test0']                                  # Initialize testing variables

    for i in range(1, 10):
        x_test = np.concatenate((x_test, data['test' + str(i)]), axis=0)
    x_test = x_test.transpose()
    y_test = []
    for i in range(10):
        y_test.extend(np.ones(data['test' + str(i)].shape[0], dtype=int)*i)
    y_test = np.asarray(y_test)

    # Keras-------------------------------------------------------------------------------------------------------------
    d = 28**2                                               # Initialize parameters
    k = 10
    n_epochs = 60
    b_size = 50

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

    fnn.fit(x_train.transpose(), y_mat_train.transpose(), batch_size=32, epochs=n_epochs)  # Train
    res = fnn.evaluate(x_test.transpose(), y_mat_test.transpose(), batch_size=b_size)          # Test

    # Plot Performance--------------------------------------------------------------------------------------------------
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

    # Plot Classified Figures-------------------------------------------------------------------------------------------
    test_images = x_test.transpose()
    test_labels = y_test
    predicted_classes = fnn.predict_classes(test_images)
    correct_indices = np.nonzero(predicted_classes == test_labels)[0]
    incorrect_indices = np.nonzero(predicted_classes != test_labels)[0]
    np.random.shuffle(correct_indices)
    np.random.shuffle((incorrect_indices))

    fig_eval1 = plt.figure(7, (14, 10))
    for i, correct in enumerate(correct_indices[:20]):
        ax = fig_eval1.add_subplot(4, 5, i + 1)
        ax.imshow(test_images[correct].reshape(28, 28), cmap='gray', interpolation='none')
        ax.set_title(
            "Predicted: {}, Truth: {}".format(predicted_classes[correct],
                                              test_labels[correct]))
        ax.axis('off')

    if save_plots:
        fig_eval1.savefig(plot_in + '/fig7',  format='pdf')

    fig_eval2 = plt.figure(8, (14, 5))
    for i, incorrect in enumerate(incorrect_indices[:10]):
        ax = fig_eval2.add_subplot(2, 5, i + 1)
        ax.imshow(test_images[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
        ax.set_title(
            "Predicted {}, Truth: {}".format(predicted_classes[incorrect],
                                             test_labels[incorrect]))
        ax.axis('off')

    if save_plots:
        fig_eval2.savefig(plot_in + '/fig8',  format='pdf')

    # Weight Visualization----------------------------------------------------------------------------------------------
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

print('FNN Accuracy is: ' + str(res[1]))

