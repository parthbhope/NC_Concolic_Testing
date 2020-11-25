# NB: do NOT import utils as this disables eager execution that seems
# to be required for proper operations of `tf.summary`.
import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

#  imports for custom tesing 
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# from tensorflow import keras
from tensorflow.keras.utils import to_categorical
# ---

default_datadir = os.getenv ('DC_DATADIR') or \
                  os.getenv ('TMPDIR', default = '/tmp') + '/sklearn_data'

image_kinds = set (('image', 'greyscale_image',))
normalized_kind = 'normalized'
unknown_kind = 'unknown'
normalized_kinds = set ((normalized_kind,))
kinds = image_kinds | normalized_kinds | set ((unknown_kind,))

choices = []

# MNIST

choices += ['mnist']
def load_mnist_data ():
    img_rows, img_cols, img_channels = 28, 28, 1
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data ()
    x_train = x_train.reshape (x_train.shape[0], img_rows, img_cols, img_channels).astype ('float32') / 255
    x_test = x_test.reshape (x_test.shape[0], img_rows, img_cols, img_channels).astype ('float32') / 255
    return (x_train, y_train), (x_test, y_test), \
           (img_rows, img_cols, img_channels), 'image', \
           [ str (i) for i in range (0, 10) ]

# Fashion-MNIST

choices += ['fashion_mnist']
def load_fashion_mnist_data ():
    img_rows, img_cols, img_channels = 28, 28, 1
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data ()
    x_train = x_train.reshape (x_train.shape[0], img_rows, img_cols, img_channels).astype ('float32') / 255
    x_test = x_test.reshape (x_test.shape[0], img_rows, img_cols, img_channels).astype ('float32') / 255
    return (x_train, y_train), (x_test, y_test), \
           (img_rows, img_cols, img_channels), 'image', \
           [ 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot' ]

# CIFAR10

choices += ['cifar10']
def load_cifar10_data ():
    img_rows, img_cols, img_channels = 32, 32, 3
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data ()
    x_train = x_train.reshape (x_train.shape[0], img_rows, img_cols, img_channels).astype ('float32') / 255
    x_test = x_test.reshape (x_test.shape[0], img_rows, img_cols, img_channels).astype ('float32') / 255
    return (x_train, y_train), (x_test, y_test), \
           (img_rows, img_cols, img_channels), 'image', \
           [ 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# adding the choice of custom dataset 
choices +=['custom']
def load_custom_data():
    print('testing for custom data and model \n')
    dataset_file = input('enter dataset 1n/2n/3n/4n/5n/6n.csv ')
    print('dataset under test :'+dataset_file)
    data = pd.read_csv('../data/'+dataset_file)
    # print(data)
    X = np.array(data.iloc[:,:-1])
    Y = np.array(data.iloc[:,-1])
    X = X.ravel()[:21952]
    print('X shape ',X.shape)
    X = np.reshape(X,(28,28,28,1))
    print('X shape new ',X.shape)
    x_train,x_test,y_train,y_test = X[:14],X[14:],Y[:14],Y[14:]
    print('random split ',x_train.shape , y_train.shape)
    # x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.5,random_state=34)
    # img_rows, img_cols, img_channels = X.shape[0], X.shape[1], 1
    # scaler = preprocessing.StandardScaler().fit(x_train)
    # scaler.transform(x_train)
    # scaler.transform(x_test)
    img_rows, img_cols, img_channels = 28,28, 1
    # x_train = np.array(x_train)
    # x_test = np.array(x_test)
    # x_train = x_train.reshape(x_train.shape[0],1, x_train.shape[1],1)
    # x_test = x_test.reshape(x_test.shape[0],1, x_test.shape[1],1)
    # print(len((X.shape[0],X.shape[1],1)))
    return (x_train,to_categorical(y_train)),(x_test,to_categorical(y_test)),(img_rows,img_cols,img_channels),'image',[0,1,2,3,4,5,6,8,9]
choices+=['custom-2']
def load_custom_2():
  # input dataset is a 2d dataset
  # the dataset is reshaped to 4 dimensions n images of 3 dimnsions (img_rows,img_cols,img_channels)
  # images chanels is set to one to make the 2d data apparently 4d
  # the model used here is mnist_complicated.h5 to check neuron coverage
  # command !python deepconcolic.py --model ../saved_models/mnist_complicated.h5  --dataset custom-2 --outputs outs/ --max-iterations 50

  print('testing for custom data and model \n')
  dataset_file = input('enter dataset 1n/2n/3n/4n/5n/6n.csv ')
  print('dataset under test :'+dataset_file)
  data = pd.read_csv('../data/'+dataset_file)
  X = np.array(data.iloc[:,:-1])
  Y = np.array(data.iloc[:,-1])
  print('original dimensions ',X.shape)
  dummy_arr = np.zeros((X.shape[0],784 - X.shape[1]))
  X = np.append(X,dummy_arr,axis=1)
  X = np.reshape(X,(X.shape[0],28,28,1))
  print('modeified dimensions  =',X.shape)
  img_rows, img_cols,img_channels = 28,28,1
  x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=34)
  return (x_train,to_categorical(y_train)),(x_test,to_categorical(y_test)),(img_rows,img_cols,img_channels),'csv',[0,1,2,3,4,5,6,8,9]


# ---

from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

openml_choices = {}
openml_choices['har'] = {
    'shuffle_last': True,
    # , 'test_size': 0.3,
    'input_kind': normalized_kind,
}

choices += ['OpenML:' + str(c) for c in openml_choices]

def load_openml_data_generic (name, datadir = default_datadir,
                              input_kind = 'unknown',
                              shuffle_last = False,
                              test_size = None):
    # print ('Retrieving OpenML dataset:', name, end = '\r', flush = True)
    ds = fetch_openml (data_home = datadir, name = name)
    # print ('Setting up', len (ds.data), 'data samples', end = '\r', flush = True)
    x_train, x_test, y_train, y_test = train_test_split (ds.data, ds.target,
                                                         test_size = test_size,
                                                         shuffle = not shuffle_last)
    if shuffle_last:
        x_train, y_train = shuffle (x_train, y_train)
        x_test, y_test = shuffle (x_test, y_test)
    labels = np.unique (ds.target)
    labl2y_dict = { y : i for i, y in enumerate (labels) }
    labl2y = np.vectorize (lambda y: labl2y_dict[y])
    y_train, y_test = labl2y (y_train), labl2y (y_test)
    # print ('Loaded', len (y_train), 'training samples, '
    #        'and', len (y_test), 'test samples')
    return (x_train, y_train.astype (int)), (x_test, y_test.astype (int)), \
           (x_train.shape[1:]), input_kind, \
           [ str (c) for c in labels ]

def load_openml_data_lambda (name):
    return lambda **kwds: load_openml_data_generic (\
        name = name, **dict (**openml_choices[name], **kwds))

# ---

def load_by_name (name, datadir = None):
    if name == 'mnist':
        return load_mnist_data ()
    elif name == 'fashion_mnist':
        return load_fashion_mnist_data ()
    elif name == 'cifar10':
        return load_cifar10_data ()
    elif name.startswith (('OpenML:', 'openml:')):
        name = name[len ('OpenML:'):]
        return load_openml_data_generic (name,
                                         **openml_choices[name],
                                         datadir = datadir)
    elif name == 'custom':
        return load_custom_data()
    elif name == 'custom-2':
      return load_custom_2()
    else:
        raise ValueError ("Unknown dataset name `{}'".format (name))
