import numpy as np
import pandas as pd

import keras
from keras import backend as K
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

def load_data(name):
    if name == "kin8nm":
        return get_kin8nm_data('./data/kin8nm')
    elif name == 'mnist':
        return get_mnist_data()
    elif name == 'MNIST-LeNet1':
        x_labels = ['conv1', 'pool1', 'conv2', 'pool2', 'fc1', 'filter_size', 'l_rate',
                    'reg_param', 'keep_prop', 'num_epoch']
        y_label = 'log_err'
        return get_surrogates_data('./data/MNIST/LeNet1_tidy.csv', x_labels, y_label)    
    else:
        raise ValueError("No such data available: {}".format(name))


def get_mnist_data():

    img_rows = 28
    img_cols = 28
    num_classes = 10
    n_train = 55000 
    n_valid = 5000
    input_shape = None

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # zero-one normalization
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train, y_train = x_train[:n_train], y_train[:n_train]
    x_valid, y_valid = x_train[-n_valid:], y_train[-n_valid:]
    x_test, y_test = x_test, y_test

    dataset = { 'num_classes': num_classes, 
            'x_train': x_train, 'y_train': y_train,
            'x_valid': x_valid, 'y_valid': y_valid, 
            'x_test': x_test, 'y_test': y_test,
            'input_shape' : input_shape 
    }
    return dataset


def get_kin8nm_data(path):
    train_file = '{}/train.csv'.format(path)
    valid_file = '{}/validation.csv'.format(path)
    test_file = '{}/test.csv'.format(path)

    print("loading data in {}...".format(path))
    try:
        train = np.loadtxt( open( train_file ), delimiter = "," )
        valid = np.loadtxt( open( valid_file ), delimiter = "," )
        test = np.loadtxt( open( test_file ), delimiter = "," )

        y_train = train[:,-1]
        y_valid = valid[:,-1]
        y_test = test[:,-1]

        x_train = train[:,0:-1]
        x_valid = valid[:,0:-1]
        x_test = test[:,0:-1]

        data = { 
                'x_train': x_train, 'y_train': y_train,
                'x_valid': x_valid, 'y_valid': y_valid, 
                'x_test': x_test, 'y_test': y_test 
        }
    except Exception as ex:
        print("Data loading failed: {}".format(ex))
        raise ValueError("Data files not found: {}".format(path))
    
    return data


def get_surrogates_data(path, x_labels, y_label):
    try:
        full_set = pd.read_csv(path)
        train, test = train_test_split(full_set, test_size=0.2)
        train, valid = train_test_split(train, test_size=0.2)
        
        x_train = train[x_labels].values
        y_train = train[y_label].values
        x_valid = valid[x_labels].values
        y_valid = valid[y_label].values
        x_test = test[x_labels].values
        y_test = test[y_label].values
        
        data = { 
                'x_train': x_train, 'y_train': y_train,
                'x_valid': x_valid, 'y_valid': y_valid, 
                'x_test': x_test, 'y_test': y_test 
        }

    except Exception as ex:
        print("Data loading failed: {}".format(ex))
        raise ValueError("Data files not found: {}".format(path))
    
    return data
