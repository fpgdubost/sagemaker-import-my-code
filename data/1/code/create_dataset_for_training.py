from keras.datasets import cifar10
import numpy as np
import scipy.misc
import random
import h5py
from pdb import set_trace as bp
import sys
import os
from basic_functions import createExpFolderandCodeList

def add_noise(image):
    noise = np.random.normal(max_x * MEAN_FACTOR_NOISE, max_x * STD_FACTOR_NOISE, [size_x, size_y, NBR_CHANNELS])
    noise[noise < 0] = 0
    image = image.astype('float64')
    image += noise
    return image

def extract_negative_samples(risk_level, sequences_x,x,y):
    nbr_neg = nbr_seq * risk_level
    x_neg_all = x[y == NEGATIVE_LABEL_ID]
    x_neg = np.zeros([nbr_neg] + list(sequences_x.shape[2:]))
    y_neg = np.zeros([nbr_neg])
    for i in range(nbr_seq):
        index = random.randrange(len(x_neg_all))
        x_neg[i] = x_neg_all[index]
    # copy sampled negatives to fill the rest
    for i in range(1, risk_level):
        x_neg[nbr_seq * i:nbr_seq * (i + 1)] = x_neg[:nbr_seq]

    return x_neg, y_neg


if __name__ == '__main__':

    MEAN_FACTOR_NOISE = 0.
    STD_FACTOR_NOISE = 0.
    NBR_CHANNELS = 3
    NEGATIVE_LABEL_ID = 9

    # exp number
    exp_number = sys.argv[1]

    # paths
    original_dataset_path = '../../experiments/815/dataset_sequences.h5'
    save_path = os.path.join('../../experiments',exp_number)

    # create exp folder
    createExpFolderandCodeList(save_path)

    # set seeds
    np.random.seed(0)
    random.seed(0)

    # load sequence dataset
    f = h5py.File(original_dataset_path, 'r')
    sequences_x_train = f['sequences_x_train']
    sequences_x_valid = f['sequences_x_valid']
    # get meta data
    nbr_images_in_seq = sequences_x_train.shape[1]
    nbr_seq = sequences_x_train.shape[0]
    size_x = sequences_x_train[0].shape[1]
    size_y = sequences_x_train[0].shape[2]
    max_x = 255

    # iterate over risk_level
    for risk_level in range(1,nbr_images_in_seq):
        # sample positives -------------------------------------------------------
        # fill x train pos
        x_train_pos = np.zeros([nbr_seq*risk_level]+list(sequences_x_train.shape[2:]))
        y_train_pos = np.ones([nbr_seq*risk_level])
        for i in range(nbr_seq):
            for j in range(risk_level):
                x_train_pos[i*risk_level+j] = sequences_x_train[i,j]
        # fill x val pos
        x_valid_pos = np.zeros([nbr_seq*risk_level]+list(sequences_x_valid.shape[2:]))
        y_valid_pos = np.ones([nbr_seq*risk_level])
        for i in range(nbr_seq):
            for j in range(risk_level):
                x_valid_pos[i*risk_level+j] = sequences_x_valid[i,j]

        # sample negatives -------------------------------------------------------
        # retrieve and split dataset for negative sampling
        (x_train_total, y_train_total), _ = cifar10.load_data()
        y_train_total = np.squeeze(y_train_total)
        split_point = int(len(x_train_total) / 2)
        x_train = x_train_total[:split_point]
        y_train = y_train_total[:split_point]
        x_valid = x_train_total[split_point:]
        y_valid = y_train_total[split_point:]

        # extract negative samples
        x_train_neg, y_train_neg = extract_negative_samples(risk_level, sequences_x_train, x_train, y_train)
        x_valid_neg, y_valid_neg = extract_negative_samples(risk_level, sequences_x_valid, x_valid, y_valid)

        # merge positives and negatives ---------------------------------------------
        x_train = np.concatenate((x_train_pos,x_train_neg),axis=0)
        y_train = np.concatenate((y_train_pos, y_train_neg), axis=0)
        x_valid = np.concatenate((x_valid_pos, x_valid_neg), axis=0)
        y_valid = np.concatenate((y_valid_pos, y_valid_neg), axis=0)

        # add noise to negative samples
        for i in range(len(x_train)):
            x_train[i] = add_noise(x_train[i])
        for i in range(len(x_valid)):
            x_valid[i] = add_noise(x_valid[i])

        # save dataset as hdf5 file
        h5f = h5py.File(os.path.join(save_path,'dataset_for_training_risk_level_'+ str(risk_level) + '.h5'), 'w')
        h5f.create_dataset('x_train', data=x_train)
        h5f.create_dataset('y_train', data=y_train)
        h5f.create_dataset('x_valid', data=x_valid)
        h5f.create_dataset('y_valid', data=y_valid)
        h5f.close()

        # visualize
        scipy.misc.imsave(os.path.join(save_path, 'train_pos.jpg'), x_train[0])
        scipy.misc.imsave(os.path.join(save_path, 'train_neg.jpg'), x_train[nbr_seq * risk_level])
        scipy.misc.imsave(os.path.join(save_path, 'valid_pos.jpg'), x_valid[0])
        scipy.misc.imsave(os.path.join(save_path, 'valid_neg.jpg'), x_valid[nbr_seq * risk_level])



    
    
    
    
    
    
    
    
