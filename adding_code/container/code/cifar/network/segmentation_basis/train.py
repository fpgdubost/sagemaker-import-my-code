import matplotlib

matplotlib.use('Agg')
from matplotlib import gridspec
import matplotlib.pyplot as plt

from keras.models import Sequential, Model, load_model
from keras.layers import Reshape, Flatten
from keras.layers import Add, Dense, Dropout, Activation, Input, GlobalAveragePooling3D, GlobalMaxPooling3D, add, \
    Lambda, Conv3D, MaxPooling3D, BatchNormalization, UpSampling3D
from keras.layers import Multiply, UpSampling2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Conv2DTranspose, \
    GlobalMaxPooling2D
from keras.layers import concatenate
from keras import callbacks
from keras import losses
from basicFunctions import createExpFolderandCodeList, ClassificationCallback, Metrics, LossHistoryBinary, LossHistory, dice_loss, compute_f1
from keras.utils import plot_model
from keras.layers import LeakyReLU
from keras import optimizers
from keras import initializers
import tensorflow as tf
import random as rn
from keras import metrics
from data_generator import DataGenerator
from shutil import copy2
import sys
import os
import numpy as np
import pandas as pd
import random
import gzip
# from sklearn.utils import class_weight
from keras.regularizers import l2
import h5py
import argparse

from tensorflow.python.framework import ops

# Paths
FINE_TUNE_PATH = None 
# Paths sagemaker
DATASET_PATH = os.path.join(os.environ.get('SM_CHANNEL_TRAINING'), 'dataset_for_training_risk_level_5.h5')
PATH_SAVE = os.environ.get('SM_MODEL_DIR')
# Paths no sagemaker
# DATASET_PATH = '/home/ubuntu/sagemaker-setup-example/data/2/dataset_for_training_risk_level_5.h5'
# PATH_SAVE = '/home/ubuntu/sagemaker-setup-example/data/3'

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--aug_shift', type=float, default=0.05)
parser.add_argument('--aug_rot', type=float, default=0.1)
parser_args, _ = parser.parse_known_args()


def get_simple_gpunet():
    print('... create model')
    nfeat = 32

    input = Input(shape=(None, None, 1))
    conv1 = Conv2D(nfeat, (3, 3), activation='relu', kernel_initializer='normal', padding='same')(input)
    conv1 = Conv2D(nfeat, (3, 3), activation='relu', kernel_initializer='normal', padding='same')(conv1)

    skip0 = concatenate([input, conv1], axis=-1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(skip0)
    conv2 = Conv2D(nfeat * 2, (3, 3), activation='relu', kernel_initializer='normal', padding='same')(pool1)
    conv2 = Conv2D(nfeat * 2, (3, 3), activation='relu', kernel_initializer='normal', padding='same')(conv2)

    skip1 = concatenate([pool1, conv2], axis=-1)
    up6 = concatenate([UpSampling2D(size=(2, 2))(skip1), skip0], axis=-1)
    conv6 = Conv2D(nfeat, (3, 3), activation='relu', kernel_initializer='normal', padding='same')(up6)

    gap = GlobalAveragePooling2D()(conv6)

    out1 = Dense(1, kernel_initializer='normal', activation='linear')(gap)
    # out1 = BatchNormalization()(out1)

    model = Model(input=input, output=out1)
    model.summary()

    # compile model
    print('compile model...')
    optimizer = 'adadelta'  # 'adadelta' 'adam' optimizers.Adam(lr=0.00001)
    model.compile(loss=dice_loss, optimizer=optimizer)

    return model


def get_unet_bn():
    print('... create model')
    nfeat = 8

    input = Input(shape=(None, None, 1))
    conv1 = Conv2D(nfeat, (3, 3), activation='relu', kernel_initializer='normal', padding='same')(input)
    conv1 = Conv2D(nfeat, (3, 3), activation='relu', kernel_initializer='normal', padding='same')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(nfeat * 2, (3, 3), activation='relu', kernel_initializer='normal', padding='same')(pool1)
    conv2 = Conv2D(nfeat * 2, (3, 3), activation='relu', kernel_initializer='normal', padding='same')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(nfeat * 4, (3, 3), activation='relu', kernel_initializer='normal', padding='same')(pool2)

    conv3 = BatchNormalization()(conv3)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    conv6 = Conv2D(nfeat, (3, 3), activation='relu', kernel_initializer='normal', padding='same')(up6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(nfeat, (3, 3), activation='relu', kernel_initializer='normal', padding='same')(up7)

    gap = GlobalAveragePooling2D()(conv7)
    gap = Dropout(dropout_rate)(gap)
    out1 = Dense(1, kernel_initializer='normal', activation='sigmoid')(gap)

    model = Model(input=input, output=out1)
    model.summary()

    # compile model
    print('compile model...')
    optimizer = 'adadelta'  # 'adadelta' 'adam' optimizers.Adam(lr=0.00001)
    model.compile(loss=dice_loss, optimizer=optimizer)

    return model


def network_classification_2D():
    nfeat = 32
    activ = 'relu'
    dropout_rate = 0.0

    input = Input(shape=(None, None, 3))

    conv1 = Conv2D(nfeat, (3, 3), activation=activ, kernel_initializer='normal', padding='same')(input)
    conv1 = Dropout(dropout_rate)(conv1)
    conv2 = Conv2D(nfeat, (3, 3), activation=activ, kernel_initializer='normal', padding='same')(conv1)
    conv2 = Dropout(dropout_rate)(conv2)
    skip0 = concatenate([input, conv2], axis=-1) 
    pool1 = MaxPooling2D(pool_size=(2, 2))(skip0)
    conv5 = Conv2D(nfeat * 2, (3, 3), activation=activ, kernel_initializer='normal', padding='same')(pool1)
    conv5 = Dropout(dropout_rate)(conv5)
    conv6 = Conv2D(nfeat * 2, (3, 3), activation=activ, kernel_initializer='normal', padding='same')(conv5)
    conv6 = Dropout(dropout_rate)(conv6)
    skip1 = concatenate([pool1, conv6], axis=-1)
    gap = GlobalAveragePooling2D()(skip1)
    gap = Dropout(dropout_rate)(gap)
    out1 = Dense(1, kernel_initializer='normal', activation='sigmoid')(gap)

    model = Model(inputs=input, outputs=out1)
    model.summary()

    # compile model
    print('compile model...')
    optimizer = optimizers.Adam(lr=parser_args.lr)  #optimizers.Adam() 'adadelta'
    loss = 'binary_crossentropy'  # dice_loss 'binary_crossentropy'
    model.compile(loss=loss, optimizer=optimizer, metrics=[compute_f1])

    return model

def network_classification_2D_deeper():
    nfeat = 64
    activ = 'relu'
    dropout_rate = 0.0

    input = Input(shape=(None, None, 1))

    conv1 = Conv2D(nfeat, (3, 3), activation=activ, kernel_initializer='normal', padding='same')(input)
    conv1 = Dropout(dropout_rate)(conv1)
    conv2 = Conv2D(nfeat, (3, 3), activation=activ, kernel_initializer='normal', padding='same')(conv1)
    conv2 = Dropout(dropout_rate)(conv2)
    conv2 = Conv2D(nfeat, (3, 3), activation=activ, kernel_initializer='normal', padding='same')(conv1)
    conv2 = Dropout(dropout_rate)(conv2)
    skip0 = concatenate([input, conv2], axis=-1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(skip0)
    conv5 = Conv2D(nfeat * 2, (3, 3), activation=activ, kernel_initializer='normal', padding='same')(pool1)
    conv5 = Dropout(dropout_rate)(conv5)
    conv6 = Conv2D(nfeat * 2, (3, 3), activation=activ, kernel_initializer='normal', padding='same')(conv5)
    conv6 = Dropout(dropout_rate)(conv6)
    conv6 = Conv2D(nfeat * 2, (3, 3), activation=activ, kernel_initializer='normal', padding='same')(conv5)
    conv6 = Dropout(dropout_rate)(conv6)
    skip1 = concatenate([pool1, conv6], axis=-1)
    gap = GlobalAveragePooling2D()(skip1)
    gap = Dropout(dropout_rate)(gap)
    out1 = Dense(1, kernel_initializer='normal', activation='sigmoid')(gap)

    model = Model(inputs=input, outputs=out1)
    model.summary()

    # compile model
    print('compile model...')
    # optimizer = 'adadelta'
    optimizer = 'adadelta'
    loss = 'binary_crossentropy'  # dice_loss 'binary_crossentropy'
    model.compile(loss=loss, optimizer=optimizer)

    return model


def network_classification_2D_bn():
    nfeat = 32
    activ = 'relu'
    dropout_rate = 0.0

    input = Input(shape=(None, None, 3))

    conv1 = Conv2D(nfeat, (3, 3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation=activ, kernel_initializer='normal', padding='same')(input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(dropout_rate)(conv1)
    conv2 = Conv2D(nfeat, (3, 3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation=activ, kernel_initializer='normal', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(dropout_rate)(conv2)
    skip0 = concatenate([input, conv2], axis=-1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(skip0)
    conv5 = Conv2D(nfeat * 2, (3, 3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation=activ, kernel_initializer='normal', padding='same')(pool1)
    conv5 = BatchNormalization()(conv5)
    conv5 = Dropout(dropout_rate)(conv5)
    conv6 = Conv2D(nfeat * 2, (3, 3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation=activ, kernel_initializer='normal', padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(dropout_rate)(conv6)
    skip1 = concatenate([pool1, conv6], axis=-1)
    gap = GlobalAveragePooling2D()(skip1)
    gap = Dropout(dropout_rate)(gap)
    out1 = Dense(1, kernel_initializer='normal', activation='sigmoid')(gap)

    model = Model(inputs=input, outputs=out1)
    model.summary()

    # compile model
    print('compile model...')
    optimizer = 'adadelta'
    loss = 'binary_crossentropy'  # dice_loss 'binary_crossentropy'
    model.compile(loss=loss, optimizer=optimizer)

    return model


def network_classification_2D_large():
    nfeat = 32
    activ = 'relu'
    dropout_rate = 0.0

    input = Input(shape=(None, None, 1))

    conv1 = Conv2D(nfeat, (3, 3), activation=activ, kernel_initializer='normal', padding='same')(input)
    conv1 = Dropout(dropout_rate)(conv1)
    conv2 = Conv2D(nfeat, (3, 3), activation=activ, kernel_initializer='normal', padding='same')(conv1)
    conv2 = Dropout(dropout_rate)(conv2)
    skip0 = concatenate([input, conv2], axis=-1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(skip0)
    conv5 = Conv2D(nfeat * 2, (3, 3), activation=activ, kernel_initializer='normal', padding='same')(pool1)
    conv5 = Dropout(dropout_rate)(conv5)
    conv6 = Conv2D(nfeat * 2, (3, 3), activation=activ, kernel_initializer='normal', padding='same')(conv5)

    conv6 = Dropout(dropout_rate)(conv6)
    skip1 = concatenate([pool1, conv6], axis=-1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(skip1)
    conv7 = Conv2D(nfeat * 4, (3, 3), activation=activ, kernel_initializer='normal', padding='same')(pool2)
    conv7 = Dropout(dropout_rate)(conv7)
    conv7 = Conv2D(nfeat * 4, (3, 3), activation=activ, kernel_initializer='normal', padding='same')(conv7)

    conv7 = Dropout(dropout_rate)(conv7)
    skip2 = concatenate([pool2, conv7], axis=-1)
    conv8 = Conv2D(nfeat * 8, (3, 3), activation=activ, kernel_initializer='normal', padding='same')(skip2)
    conv8 = Dropout(dropout_rate)(conv8)
    conv8 = Conv2D(nfeat * 8, (3, 3), activation=activ, kernel_initializer='normal', padding='same')(conv8)

    conv8 = Dropout(dropout_rate)(conv8)
    skip3 = concatenate([pool2, skip2, conv8], axis=-1)
    gap = GlobalAveragePooling2D()(skip3)
    gap = Dropout(dropout_rate)(gap)
    out1 = Dense(1, kernel_initializer='normal', activation='sigmoid')(gap)

    model = Model(inputs=input, outputs=out1)
    model.summary()

    # compile model
    print('compile model...')
    # optimizer = 'adadelta'
    optimizer = optimizers.Adam()
    loss = 'binary_crossentropy'  # dice_loss 'binary_crossentropy'
    model.compile(loss=loss, optimizer=optimizer)

    return model


def network_regression():
    nfeat = 32
    activ = 'relu'
    dropout_rate = 0.0

    input = Input(shape=(None, None, None, 1))

    conv1 = Conv3D(nfeat, (3, 3, 3), activation=activ, kernel_initializer='normal', padding='same')(input)
    conv1 = Dropout(dropout_rate)(conv1)
    conv2 = Conv3D(nfeat, (3, 3, 3), activation=activ, kernel_initializer='normal', padding='same')(conv1)
    conv2 = Dropout(dropout_rate)(conv2)
    skip0 = concatenate([input, conv2], axis=-1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(skip0)
    conv5 = Conv3D(nfeat * 2, (3, 3, 3), activation=activ, kernel_initializer='normal', padding='same')(pool1)
    conv5 = Dropout(dropout_rate)(conv5)
    conv6 = Conv3D(nfeat * 2, (3, 3, 3), activation=activ, kernel_initializer='normal', padding='same')(conv5)
    conv6 = Dropout(dropout_rate)(conv6)
    skip1 = concatenate([pool1, conv6], axis=-1)
    gap = GlobalAveragePooling3D()(skip1)
    gap = Dropout(dropout_rate)(gap)
    out1 = Dense(1, kernel_initializer='normal', activation='linear')(gap)

    model = Model(inputs=input, outputs=out1)
    model.summary()

    # compile model
    print('compile model...')
    optimizer = 'adadelta'
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def get_densenet():
    model = DenseNet(nb_dense_block=2, classes=1)
    model.summary()

    # compile model
    print('compile model...')
    optimizer = optimizers.Adam()
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def get_list_sets(list_folder, isTest=True):
    files_training = pd.read_csv(os.path.join(list_folder, 'train.csv'), header=None)
    files_valid = pd.read_csv(os.path.join(list_folder, 'valid.csv'), header=None)
    files_testing = None
    if isTest:
        files_testing = pd.read_csv(os.path.join(list_folder, 'test.csv'), header=None)

    return files_training, files_valid, files_testing


def save_sets_id(path_save, files_training, files_valid, files_testing=None, savePaths=None):
    files_training.to_csv(os.path.join(path_save, 'train.csv'), header=False, index=False)
    files_valid.to_csv(os.path.join(path_save, 'valid.csv'), header=False, index=False)
    if files_testing != None:
        files_testing.to_csv(os.path.join(path_save, 'test.csv'), header=False, index=False)


def create_set(listFiles, normalization='minMax'):
    # find shape
    shape = nipy.load_image(os.path.join(path_data, listFiles[0] + extension_data))._data.shape

    # load data
    # create emlpty arrays
    nbr_examples = len(listFiles)
    X = np.zeros([nbr_examples] + shape, dtype=np.float32)
    Y = np.zeros([nbr_examples] + shape, dtype=np.int)
    # iterate over files and fill arrays
    for i, filename in enumerate(listFiles):
        # load scans
        print('file {}/{} : {}'.format(i + 1, len(listFiles), filename))
        data = nipy.load_image(os.path.join(path_data, filename + extension_data))._data
        label = nipy.load_image(os.path.join(path_gt, filename + extension_gt))._data

        # normalize
        if np.count_nonzero(data) > 0:
            if normalization == 'minMax':
                minData = np.amin(data)
                maxData = np.amax(data)
                data = (data - minData) * 1. / (maxData - minData)
            elif normalization == 'percentile':
                minData = np.percentile(data, 1)
                maxData = np.percentile(data, 99)
                data = (data - minData) * 1. / (maxData - minData)
            elif normalization == 'meanstd':
                std_data = np.std(data)
                data = (data - np.mean(data)) * 1. / std_data

                # store in X
        X[i] = np.array(data)
        Y[i] = np.array(label)

    X = np.expand_dims(X, axis=-1)
    Y = np.expand_dims(Y, axis=-1)

    return X, Y


def shuffle_data(data, labels):
    indexes_shuffled = list(range(len(data)))
    random.shuffle(indexes_shuffled)
    data_shuffled = data[indexes_shuffled]
    labels_shuffled = labels[indexes_shuffled]

    return data_shuffled, labels_shuffled


def normalize(data, normalization):
    if np.count_nonzero(data) > 0:
        if normalization == 'minMax':
            min_data = np.amin(data)
            max_data = np.amax(data)
            data_normalized = (data - min_data) * 1. / (max_data - min_data)
        elif normalization == 'percentile':
            min_data = np.percentile(data, 1)
            max_data = np.percentile(data, 99)
            data_normalized = (data - min_data) * 1. / (max_data - min_data)
        elif normalization == 'meanstd':
            std_data = np.std(data)
            data_normalized = (data - np.mean(data)) * 1. / std_data

        return data_normalized


def create_array_from_list_paths(path_data_list):
    # load data
    array_list = []
    label_list = []
    for path_data in path_data_list:
        # load arrays
        f = gzip.GzipFile(os.path.join(path_data, 'video_corr_feat.npy.gz'), "r")
        array_list.append(np.load(f))
        # load labels
        label_list.append(pd.read_csv(os.path.join(path_data, 'segment_labels.csv'))['label'].values)

    # precompute size of array
    total_nbr_samples = 0
    for array in array_list:
        total_nbr_samples += array.shape[0]

    # define empty array
    full_array = np.zeros([total_nbr_samples] + list(array_list[0].shape[1:]))
    full_labels = np.zeros([total_nbr_samples] + list(label_list[0].shape[1:]))

    # fill full array
    idx = 0
    for i, array in enumerate(array_list):
        nbr_samples_array = array.shape[0]
        # fill array
        full_array[idx:idx + nbr_samples_array] = array
        # fill labels
        full_labels[idx:idx + nbr_samples_array] = label_list[i]
        # update index
        idx += nbr_samples_array

    return full_array, full_labels

def get_list_path_dataset(list_experiment_ids):

    list_training_set_paths = [os.path.join('/share/pi/cleemess/fdubost/eeg_video/experiments',str(ID)) for ID in list_experiment_ids]

    return list_training_set_paths

def prepare_single_dataset(dataset_path,split):
    f = h5py.File(dataset_path, 'r')
    x = f['x_'+str(split)]
    y = f['y_'+str(split)]
    # X
    x = np.copy(x)
    # normalize X
    for i in range(len(x)):
        x[i] = normalize(x[i], normalization)
    # Y
    y = np.expand_dims(y, axis=-1)

    return x,y

def load_and_prepare_data(dataset_path):

    # load data
    train_set_x, train_set_y = prepare_single_dataset(dataset_path,'train')
    valid_set_x, valid_set_y = prepare_single_dataset(dataset_path,'valid')

    # Coroflo generator
    datagen = DataGenerator(train_set_x, train_set_y, params, paddingGT=padding, batch_size=batch_size,
                            shuffle=True, plotgenerator=5)

    # visualize output of generator
    for t in range(5):
        # transform the data
        X_aug, Y_aug = datagen.prepare_batch(train_set_x[:10], train_set_y[:10])
        if X_aug.shape == Y_aug.shape:  # if mask labels
            datagen.save_images(X_aug, Y_aug, train_set_x, train_set_y)
        else:  # if image level labels
            datagen.save_images(X_aug, None, train_set_x, train_set_y, 'viridis')

    return datagen, train_set_x, train_set_y, valid_set_x, valid_set_y


def train_model():
    # create folder of experiments code, parameters and results
    createExpFolderandCodeList(PATH_SAVE)

    # fix seeds
    if fixedSeed:
        tfseed = fixSeeds()
    else:
        tfseed = None

    # create DL model
    model = network_classification_2D()  # get_gpunet small3Dresnet(isRegression) get_simple_gpunet(isRegression) get_attention_net(isRegression) get_gpunet(isRegression)

    # load model for fine-tuning
    if FINE_TUNE_PATH != None:
        model = load_model(os.path.join(FINE_TUNE_PATH, 'best_weights.hdf5'), custom_objects={ 'compute_f1': compute_f1 })  # load model (with optimizer state)

    # save model
    print('save model...')
    model_json = model.to_json()
    # plot_model(model, show_shapes=True, to_file=os.path.join(path_save, 'model.png'))
    with open(os.path.join(PATH_SAVE, 'model.json'), "w") as json_file:
        json_file.write(model_json)

    # load dataset
    datagen, train_set_x, train_set_y, valid_set_x, valid_set_y = load_and_prepare_data(DATASET_PATH)

    # Calculate the weights for each class so that we can balance the data
    # weights = class_weight.compute_class_weight('balanced', np.unique(train_set_y), np.squeeze(train_set_y))
    # weights = weights

    # callbacks (to be executed after each epoch)
    best_weights_filepath = os.path.join(PATH_SAVE, 'best_weights.hdf5')
    saveBestModel = callbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss',
                                              verbose=1, save_best_only=True, mode='auto')

    # train model
    print('training ...')
    step_per_epoch = 100 #len(train_set_y) / batch_size
    model.fit(datagen.flow(train_set_x, train_set_y, batch_size=batch_size, shuffle=True),
                        steps_per_epoch=step_per_epoch, epochs=nb_epoch,
                        verbose=1, validation_data=(valid_set_x, valid_set_y), callbacks=[saveBestModel])


def fixSeeds():
    # NUMPY
    np.random.seed(42)

    # TENSOFRFLOW
    # Clears the default graph stack and resets the global default graph.
    ops.reset_default_graph()
    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tfseed = 42
    if tf.__version__.split('.')[0] == '1':
        tf.set_random_seed(tfseed)
    elif tf.__version__.split('.')[0] == '2':
        tf.random.set_seed(tfseed)
    # PYTHON
    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    os.environ['PYTHONHASHSEED'] = '0'

    # RANDOM
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(42)

    return tfseed


if __name__ == '__main__':
    fixedSeed = False

    # normalization
    normalization = 'minMax'  # minMax percentile meanstd

    # data augmentation parameters
    # creating a single dictionary for parameters
    params = {}
    # 0 : affine tranform, 1 : pixel-wise elastic deformation, 2 : grid-wise elastic deformation
    params["augmentation"] = [0, 0, 0]
    # intensities
    params["intensity_shift"] = False
    params["intensity_shift_params"] = 3
    # initalize parameter objects
    augmentparams = dict()
    params["random_deform"] = dict()
    params["only"] = None
    params["e_deform_p"] = dict()
    params["e_deform_g"] = dict()
    # Standard data augmentation
    shift = parser_args.aug_shift  # 0.2
    rotation = parser_args.aug_rot  # 0.3
    params["random_deform"]['width_shift_range'] = shift
    params["random_deform"]['height_shift_range'] = shift
    params["random_deform"]['depth_shift_range'] = shift
    params["random_deform"]['rotation_range_alpha'] = rotation
    params["random_deform"]['rotation_range_beta'] = rotation
    params["random_deform"]['rotation_range_gamma'] = rotation
    params["random_deform"]['horizontal_flip'] = True
    params["random_deform"]['vertical_flip'] = True
    params["random_deform"]['z_flip'] = False
    # Add elastic deformations
    # pixel def
    params["e_deform_p"]["alpha"] = 3
    params["e_deform_p"]["sigma"] = 2
    # grid def
    params["e_deform_g"]["points"] = 3
    params["e_deform_g"]["sigma"] = 2
    # define saveFolder
    params["savefolder"] = PATH_SAVE
    # padding
    padding = 0

    # training parameters
    batch_size = 1  # 64
    nb_epoch = 2000  # 2000 400 150
    dropout_rate = 0.0  # 0.3

    train_model()
