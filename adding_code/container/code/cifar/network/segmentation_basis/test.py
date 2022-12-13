import matplotlib
from keras.datasets import cifar10

matplotlib.use('Agg')
from matplotlib import pyplot as plt

import sys
import numpy as np
import os
from keras.models import model_from_json
from attentionComputation import getCAM, getGradCAM, getGradOrGuidedBackprop
import gzip
from basicFunctions import createExpFolderandCodeList
from pdb import set_trace as bp
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def boostrapping_CI(custom_metric, data, nbr_runs=1000):
    # Confidence Interval Estimation of an ROC Curve: An Application of Generalized Half Normal and Weibull Distributions
    nbr_scans = len(data.index)

    list_metric = []
    # compute mean
    for r in range(nbr_runs):
        # sample random indexes
        ind = np.random.randint(nbr_scans, size=nbr_scans)

        # select random subset
        data_bootstrapped = data.iloc[ind]

        # compute metrics
        metric = custom_metric(data_bootstrapped)
        list_metric.append(metric)

    # store variable in dictionary
    metric_stats = {}
    metric_stats['avg_metric'] = round(np.average(list_metric),nbr_digits)
    metric_stats['metric_ci_lb'] = round(np.percentile(list_metric, 5),nbr_digits)
    metric_stats['metric_ci_ub'] = round(np.percentile(list_metric, 95),nbr_digits)

    return metric_stats

def f1_boot(data):
    val_target = data['y_true']
    val_predict = data['y_pred']
    tp, tn, fp, fn = compute_tptnfpfn(val_target, val_predict)
    f1 = compute_f1(tp, tn, fp, fn)

    return f1

def acc_boot(data):
    val_target = data['y_true']
    val_predict = data['y_pred']
    tp, tn, fp, fn = compute_tptnfpfn(val_target, val_predict)
    acc = compute_accuracy(tp, tn, fp, fn)

    return acc

def prec_boot(data):
    val_target = data['y_true']
    val_predict = data['y_pred']
    tp, tn, fp, fn = compute_tptnfpfn(val_target, val_predict)
    prec = compute_precision(tp, tn, fp, fn)

    return prec

def rec_boot(data):
    val_target = data['y_true']
    val_predict = data['y_pred']
    tp, tn, fp, fn = compute_tptnfpfn(val_target, val_predict)
    rec = compute_recall(tp, tn, fp, fn)

    return rec

def auc_boot(data):
    val_target = data['y_true']
    val_predict = data['y_pred']
    auc = roc_auc_score(val_target, val_predict)

    return auc


def ap_boot(data):
    val_target = data['y_true']
    val_predict = data['y_pred']
    ap = average_precision_score(val_target, val_predict)

    return ap


def compute_all_metrics(val_target, val_predict):
    tp, tn, fp, fn = compute_tptnfpfn(val_target, val_predict)
    f1 = round(compute_f1(tp, tn, fp, fn), nbr_digits)
    recall = round(compute_recall(tp, tn, fp, fn), nbr_digits)
    precis = round(compute_precision(tp, tn, fp, fn), nbr_digits)
    acc = round(compute_accuracy(tp, tn, fp, fn), nbr_digits)

    return f1, recall, precis, acc


def compute_tptnfpfn(val_target, val_predict):
    # cast to boolean
    val_target = val_target.astype('bool')
    val_predict = val_predict.astype('bool')

    tp = np.count_nonzero(val_target * val_predict)
    tn = np.count_nonzero(~val_target * ~val_predict)
    fp = np.count_nonzero(~val_target * val_predict)
    fn = np.count_nonzero(val_target * ~val_predict)

    return tp, tn, fp, fn


def compute_f1(tp, tn, fp, fn):
    f1 = tp * 1. / (tp + 0.5 * (fp + fn) + sys.float_info.epsilon)
    return f1


def compute_recall(tp, tn, fp, fn):
    recall = tp * 1. / (tp + fn + sys.float_info.epsilon)
    return recall


def compute_precision(tp, tn, fp, fn):
    precision = tp * 1. / (tp + fp + sys.float_info.epsilon)
    return precision


def compute_accuracy(tp, tn, fp, fn):
    acc = (tp + tn) * 1. / (tp + tn + fp + fn + sys.float_info.epsilon)
    return acc


def load_model(CNN_arch,CNN_weights,custom_objects=None):
    
    #load architecture
    json_file = open(CNN_arch, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects=custom_objects)
    
    #laod weights
    model.load_weights(CNN_weights)
       
    return model

def add_noise(image):
    # add noise
    noise = np.random.normal(max_x * MEAN_FACTOR_NOISE, max_x * STD_FACTOR_NOISE, [size_x, size_y])
    noise[noise < 0] = 0
    image = image.astype('float64')
    image += noise
    return image

if __name__ == '__main__':

    # noise parameters
    MEAN_FACTOR_NOISE = 0.
    STD_FACTOR_NOISE = 0.

    # inputs
    EXPERIMENT_ID = sys.argv[1]
    MODEL_EXP_ID = sys.argv[2]

    # normalization
    normalization = 'minMax'

    # treshold
    thresh = 0.5

    # nbr digits to display in metrics
    nbr_digits = 3

    # paths
    save_path = os.path.join('../../../experiments', EXPERIMENT_ID)
    model_path = os.path.join('../../../experiments/', MODEL_EXP_ID)
    CNN_weights = os.path.join(model_path,'best_weights.hdf5')
    CNN_arch = os.path.join(model_path,'model.json')

    # create folder of experiments code, parameters and results
    createExpFolderandCodeList(save_path)

    # load model
    model = load_model(CNN_arch, CNN_weights)

    # load test set
    _, (data, y_true) = cifar10.load_data()
    y_true = np.squeeze(y_true)

    # select only zeros and ones
    data = data[(y_true == 0) + (y_true == 1)]
    y_true = y_true[(y_true == 0) + (y_true == 1)]

    # get meta data
    size_x = data[0].shape[-2]
    size_y = data[0].shape[-1]
    max_x = np.max(data)

    # add noise
    for i in range(len(data)):
        data[i] = add_noise(data[i])

    #normalize
    if np.count_nonzero(data) > 0:
        if normalization == 'minMax':
            minData = np.amin(data)
            maxData = np.amax(data)
            data = (data - minData)*1. / (maxData - minData) 
        elif normalization == 'percentile':
            minData = np.percentile(data,1)
            maxData = np.percentile(data,99)
            data = (data - minData)*1. / (maxData - minData)        
        elif normalization == 'meanstd':
            std_data = np.std(data)
            data = (data - np.mean(data))*1. / std_data

    #predict
    prediction = np.squeeze(model.predict(data, batch_size=1, verbose = 0))
    if thresh:
        prediction = np.where(prediction > 0.5, 1, 0)

    # create and fill dataframe
    results = pd.DataFrame()
    for i in range(len(prediction)):
        results = results.append({'y_true':y_true[i], 'y_pred':prediction[i]},ignore_index=True)
    # save dataframe
    results.to_csv(os.path.join(save_path,'predictions_and_gt.csv'))

    # compute metrics
    f1, recall, precis, acc = compute_all_metrics(y_true,prediction)
    auc = round(roc_auc_score(y_true,prediction),nbr_digits)
    ap = round(average_precision_score(y_true,prediction),nbr_digits)
    # compute CI
    metric_stats_f1 = boostrapping_CI(f1_boot, results)
    metric_stats_acc = boostrapping_CI(acc_boot, results)
    metric_stats_prec = boostrapping_CI(prec_boot, results)
    metric_stats_rec = boostrapping_CI(rec_boot, results)
    metric_stats_auc = boostrapping_CI(auc_boot, results)
    metric_stats_ap = boostrapping_CI(ap_boot, results)
    # save metrics
    metrics = pd.DataFrame()
    metrics = metrics.append({'stat':'average','f1': f1,'recall':recall,'precis':precis,'acc':acc,'auc':auc,'ap':ap},ignore_index=True)
    metrics = metrics.append({'stat':'lower bound','f1': metric_stats_f1['metric_ci_lb'],'recall':metric_stats_rec['metric_ci_lb'],
                              'precis':metric_stats_prec['metric_ci_lb'],'acc':metric_stats_acc['metric_ci_lb'],
                              'auc':metric_stats_auc['metric_ci_lb'],'ap':metric_stats_ap['metric_ci_lb']},ignore_index=True)
    metrics = metrics.append({'stat':'upper bound','f1': metric_stats_f1['metric_ci_ub'], 'recall': metric_stats_rec['metric_ci_ub'],
                              'precis': metric_stats_prec['metric_ci_ub'], 'acc': metric_stats_acc['metric_ci_ub'],
                              'auc': metric_stats_auc['metric_ci_ub'], 'ap': metric_stats_ap['metric_ci_ub']},
                             ignore_index=True)
    metrics.to_csv(os.path.join(save_path,'metrics.csv'))




    
    


    
    
