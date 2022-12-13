from keras.datasets import cifar10
import numpy as np
import scipy.misc
import random
import h5py
from pdb import set_trace as bp
import sys
import os
import imageio
from basic_functions import createExpFolderandCodeList

def create_list_sequences(x, y):
    # extract only 0 and 1 images
    x_0 = x[y == NEGATIVE_LABEL_ID]
    x_1 = x[y == POSITIVE_LABEL_ID]

    # create empty arrays
    sequences_x = np.zeros([NBR_SEQ,NBR_IMAGES_IN_SEQ,size_x,size_y,NBR_CHANNELS])
    sequences_y = np.zeros([NBR_SEQ,NBR_IMAGES_IN_SEQ])

    # fill the arrays
    for i in range(NBR_SEQ):
        # draw at random the length of the positive series
        length_positive_series = random.randint(0,NBR_IMAGES_IN_SEQ)
        for j in range(NBR_IMAGES_IN_SEQ):
            if j < length_positive_series:
                # draw a random 1 image
                random_1_image = x_1[random.randint(0, x_1.shape[0]-1)]
                # fill array
                sequences_x[i, j] = random_1_image
                sequences_y[i, j] = 1
            else:
                # draw a random 0 image
                random_0_image = x_0[random.randint(0, x_0.shape[0]-1)]
                # fill array
                sequences_x[i,j] = random_0_image
                sequences_y[i,j] = 0

    return sequences_x, sequences_y

if __name__ == '__main__':

    NBR_IMAGES_IN_SEQ = 10
    NBR_SEQ = 50
    NBR_CHANNELS = 3
    POSITIVE_LABEL_ID = 1
    NEGATIVE_LABEL_ID = 9

    # exp number
    exp_number = '1' #sys.argv[1]

    # paths
    save_path = os.path.join('../../../../../data',exp_number)

    # create exp folder
    createExpFolderandCodeList(save_path)

    # set seeds
    np.random.seed(0)
    random.seed(0)
        
    # retrieve and split dataset
    (x_train_total, y_train_total), _ = cifar10.load_data()
    y_train_total = np.squeeze(y_train_total)
    split_point = int(len(x_train_total)/2)
    x_train = x_train_total[:split_point]
    y_train = y_train_total[:split_point]
    x_valid = x_train_total[split_point:]
    y_valid = y_train_total[split_point:]

    # get sizes
    size_x = x_train[0].shape[0]
    size_y = x_train[0].shape[1]

    # create list of sequence as save the images and labels in two arrays
    # image array of size NBR_IMAGES_IN_SEQ * NBR_SEQ * 28 * 28
    # label array of size NBR_IMAGES_IN_SEQ * NBR_SEQ
    sequences_x_train, sequences_y_train = create_list_sequences(x_train, y_train)
    sequences_x_valid, sequences_y_valid = create_list_sequences(x_valid, y_valid)

    # save dataset as hdf5 file
    h5f = h5py.File(os.path.join(save_path,'dataset_sequences.h5'), 'w')
    h5f.create_dataset('sequences_x_train', data=sequences_x_train)
    h5f.create_dataset('sequences_y_train', data=sequences_y_train)
    h5f.create_dataset('sequences_x_valid', data=sequences_x_valid)
    h5f.create_dataset('sequences_y_valid', data=sequences_y_valid)
    h5f.close()


    # visualize sequence
    for seq_id in range(NBR_SEQ):
        # fill image
        image = np.zeros([size_y,size_x*NBR_IMAGES_IN_SEQ,NBR_CHANNELS])
        for i in range(NBR_IMAGES_IN_SEQ):
            image[0:size_y,i*size_x:(i+1)*size_x] = sequences_x_train[seq_id,i]

        # save image
        imageio.imwrite(os.path.join(save_path,'seq_1_'+str(seq_id)+'.jpg'), image.astype(np.uint8))

    
    
    
    
    
    
    
    
    

    
    
