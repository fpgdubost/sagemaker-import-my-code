import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os
import json
import csv
import cv2
import shutil
import sys
import seaborn as sns
import pandas as pd
import math
import skimage.transform
import random

from scipy.signal import correlate
import gzip
import pandas as pd
from ipdb import set_trace as bp

import numpy as np
from PIL import Image
import glob

from basicFunctions import createExpFolderandCodeList


def extract_and_save_individual_samples(path_array):
    # load full set
    f = gzip.GzipFile(path_array, "r")
    full_set = np.load(f)

    # select target sample
    sample = full_set[sample_id]

    # save segment cc_array as numpy array
    corr_feat_savepath = os.path.join(root_savepath, 'single_sample.npy.gz')
    f = gzip.GzipFile(corr_feat_savepath, "w")
    np.save(file=f, arr=sample)
    f.close()

    # save video name and path
    df = pd.DataFrame(columns=['data location'])
    df = df.append({'data location':input_exp_path},ignore_index=True)
    df.to_csv(os.path.join(root_savepath, 'info.csv'),index=False)

if __name__ == "__main__":
    # usage
    # python script.py experiment_id

    # retrieve video filepath
    input_exp_path = sys.argv[2]
    path_array = glob.glob(os.path.join(input_exp_path,'*.npy.gz'))[0]

    #retrieve sampled id
    sample_id = int(sys.argv[3])

    # parse path
    experiment_id = sys.argv[1]
    root_savepath = os.path.join('/share/pi/cleemess/fdubost/eeg_video/experiments', experiment_id)
    # create exp folder
    createExpFolderandCodeList(root_savepath)

    # launch processing
    extract_and_save_individual_samples(path_array)



