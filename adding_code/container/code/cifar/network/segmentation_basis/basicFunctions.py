import matplotlib

matplotlib.use('Agg')
import os
from keras.models import model_from_json
from keras import backend as K
import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
import csv
from keras import callbacks
from keras import metrics
from math import floor
import scipy.misc
import math
import random
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator
import sys
from shutil import copy2
import pandas as pd

def createExpFolderandCodeList(save_path, files=[]):
    # result folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    code_folder_path = os.path.join(save_path, 'code')
    if not os.path.exists(code_folder_path):
        os.makedirs(code_folder_path)
    # save code files
    for file_name in os.listdir('.') + files:
        if not os.path.isdir(file_name):
            copy2(file_name, os.path.join(save_path, 'code', file_name))

def dice(y_true, y_pred):
    smoothing_factor = 10 ** -3
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smoothing_factor) / (K.sum(y_true_f) + K.sum(y_pred_f) + smoothing_factor)

def dice_loss(y_true, y_pred):
    return 1. - dice(y_true, y_pred)

def compute_f1(true,pred):
    ground_positives = K.sum(true, axis=0)  # = TP + FN
    pred_positives = K.sum(pred, axis=0)  # = TP + FP
    true_positives = K.sum(true * pred, axis=0)  # = TP

    precision = true_positives / (pred_positives + K.epsilon())
    recall = true_positives / (ground_positives + K.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return -f1

def compute_tptnfpfn(val_target, val_predict):
    # cast to boolean
    val_target = val_target.astype('bool')
    val_predict = val_predict.astype('bool')

    tp = np.count_nonzero(val_target * val_predict)
    tn = np.count_nonzero(~val_target * ~val_predict)
    fp = np.count_nonzero(~val_target * val_predict)
    fn = np.count_nonzero(val_target * ~val_predict)

    return tp, tn, fp, fn


def compute_f1_back(val_target, val_predict):
    tp, tn, fp, fn = compute_tptnfpfn(val_target, val_predict)
    f1 = tp * 1. / (tp + 0.5 * (fp + fn) + sys.float_info.epsilon)
    return f1


def compute_recall(val_target, val_predict):
    tp, tn, fp, fn = compute_tptnfpfn(val_target, val_predict)
    recall = tp * 1. / (tp + fn + sys.float_info.epsilon)
    return recall


def compute_precision(val_target, val_predict):
    tp, tn, fp, fn = compute_tptnfpfn(val_target, val_predict)
    precision = tp * 1. / (tp + fp + sys.float_info.epsilon)
    return precision


def compute_accuracy(val_target, val_predict):
    tp, tn, fp, fn = compute_tptnfpfn(val_target, val_predict)
    acc = (tp + tn) * 1. / (tp + tn + fp + fn + sys.float_info.epsilon)
    return acc


class Metrics(callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs):
        # 5.4.1 For each validation batch
        for batch_index in range(0, len(self.validation_data[0])):
            # 5.4.1.1 Get the batch target values
            temp_target = self.validation_data[1][batch_index]
            # 5.4.1.2 Get the batch prediction values
            temp_predict = (np.asarray(self.model.predict(np.expand_dims(
                self.validation_data[0][batch_index], axis=0)))).round()
            # 5.4.1.3 Append them to the corresponding output objects
            if batch_index == 0:
                val_target = temp_target
                val_predict = temp_predict
            else:
                val_target = np.vstack((val_target, temp_target))
                val_predict = np.vstack((val_predict, temp_predict))

        tp, tn, fp, fn = self.compute_tptnfpfn(val_target, val_predict)
        val_f1 = round(self.compute_f1(tp, tn, fp, fn), 4)
        val_recall = round(self.compute_recall(tp, tn, fp, fn), 4)
        val_precis = round(self.compute_precision(tp, tn, fp, fn), 4)

        self.val_f1s.append(val_f1)
        self.val_recalls.append(val_recall)
        self.val_precisions.append(val_precis)

        # Add custom metrics to the logs, so that we can use them with
        # EarlyStop and csvLogger callbacks
        logs["val_f1"] = val_f1
        logs["val_recall"] = val_recall
        logs["val_precis"] = val_precis

        return

    def compute_tptnfpfn(self, val_target, val_predict):
        # cast to boolean
        val_target = val_target.astype('bool')
        val_predict = val_predict.astype('bool')

        tp = np.count_nonzero(val_target * val_predict)
        tn = np.count_nonzero(~val_target * ~val_predict)
        fp = np.count_nonzero(~val_target * val_predict)
        fn = np.count_nonzero(val_target * ~val_predict)

        return tp, tn, fp, fn

    def compute_f1(self, tp, tn, fp, fn):
        f1 = tp * 1. / (tp + 0.5 * (fp + fn) + sys.float_info.epsilon)
        return f1

    def compute_recall(self, tp, tn, fp, fn):
        recall = tp * 1. / (tp + fn + sys.float_info.epsilon)
        return recall

    def compute_precision(self, tp, tn, fp, fn):
        precision = tp * 1. / (tp + fp + sys.float_info.epsilon)
        return precision

    def compute_accuracy(self, tp, tn, fp, fn):
        acc = (tp + tn) * 1. / (tp + tn + fp + fn + sys.float_info.epsilon)
        return acc


class ClassificationCallback(callbacks.Callback):

    def __init__(self, savePaths):
        # path to save csv table of metrics
        self.savePath = savePaths

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs):
        # predict values
        # 5.4.1 For each validation batch
        for batch_index in range(0, len(self.validation_data[0])):
            # 5.4.1.1 Get the batch target values
            temp_target = self.validation_data[1][batch_index]
            # 5.4.1.2 Get the batch prediction values
            temp_predict = (np.asarray(self.model.predict(np.expand_dims(
                self.validation_data[0][batch_index], axis=0)))).round()
            # 5.4.1.3 Append them to the corresponding output objects
            if batch_index == 0:
                val_target = temp_target
                val_predict = temp_predict
            else:
                val_target = np.vstack((val_target, temp_target))
                val_predict = np.vstack((val_predict, temp_predict))

        # compute metrics
        tp, tn, fp, fn = self.compute_tptnfpfn(val_target, val_predict)
        val_f1 = round(self.compute_f1(tp, tn, fp, fn), 4)
        val_recall = round(self.compute_recall(tp, tn, fp, fn), 4)
        val_precision = round(self.compute_precision(tp, tn, fp, fn), 4)
        val_acc = round(self.compute_accuracy(tp, tn, fp, fn), 4)

        # write on disk
        path_csv = os.path.join(self.savePath, 'evolution.csv')
        # write header with column names
        if epoch == 0:
            self.write_names_csv(['loss', 'val_loss', 'f1', 'recall', 'precision', 'accuracy', 'fp'], path_csv)
        # update current values
        self.write_csv([logs['loss'], logs['val_loss'], val_f1, val_recall, val_precision, val_acc, fp], path_csv)

        # Add custom metrics to the logs, so that we can use them with
        # EarlyStop and csvLogger callbacks
        self.val_f1s.append(val_f1)
        self.val_recalls.append(val_recall)
        self.val_precisions.append(val_precision)
        # add to logs
        logs["val_f1"] = val_f1
        logs["val_recall"] = val_recall
        logs["val_precision"] = val_precision

        # plot the validation curve
        val_loss_all = pd.read_csv(path_csv,sep=';')["val_loss"]
        plt.figure()
        plt.plot(val_loss_all)
        plt.savefig(os.path.join(self.savePath, 'val_loss.jpg'))
        plt.close()

        return

    def compute_tptnfpfn(self, val_target, val_predict):
        # cast to boolean
        val_target = val_target.astype('bool')
        val_predict = val_predict.astype('bool')

        tp = np.count_nonzero(val_target * val_predict)
        tn = np.count_nonzero(~val_target * ~val_predict)
        fp = np.count_nonzero(~val_target * val_predict)
        fn = np.count_nonzero(val_target * ~val_predict)

        return tp, tn, fp, fn

    def compute_f1(self, tp, tn, fp, fn):
        f1 = tp * 1. / (tp + 0.5 * (fp + fn) + sys.float_info.epsilon)
        return f1

    def compute_recall(self, tp, tn, fp, fn):
        recall = tp * 1. / (tp + fn + sys.float_info.epsilon)
        return recall

    def compute_precision(self, tp, tn, fp, fn):
        precision = tp * 1. / (tp + fp + sys.float_info.epsilon)
        return precision

    def compute_accuracy(self, tp, tn, fp, fn):
        acc = (tp + tn) * 1. / (tp + tn + fp + fn + sys.float_info.epsilon)
        return acc

    def write_csv(self, losses, path_write):
        with open(path_write, "a") as my_file:
            writer = csv.writer(my_file, delimiter=';')
            writer.writerow(losses)  # should be a list

    def write_names_csv(self, names, path_write):
        with open(path_write, "w") as my_file:
            writer = csv.writer(my_file, delimiter=';')
            writer.writerow(names)



class LossHistory(callbacks.Callback):

    def __init__(self, savePaths):

        # parameter of the csv and plot
        self.reports = [{'type': 'text',
                         'file': 'evolution.csv',
                         'outputs': [0],
                         'order': ['metric', 'set', 'output']},
                        {'type': 'plot',
                         'file': 'plot.png',
                         'outputs': [0],
                         'order': ['metric', 'set', 'output']}]

        self.savePaths = savePaths

        metrics = []

        self.dimSpecs = {'set': ['', 'val_'], 'output': [0], 'metric': ['loss'] + metrics}

        outputs = ['']

        #        addVars = [{'name': 'loss', 'kerasName': 'loss', 'val': []},
        #                    {'name': 'val_loss', 'kerasName': 'val_loss', 'val': []}]

        addVars = []

        #        addNesting = [('overall loss', [('overall loss', [0, 1])])]

        addNesting = []

        def getKerasName(v):

            return '%s%s%s' % (v['set'], outputs[v['output']], v['metric'])

        for r in self.reports:

            # list of variables to monitor
            rvars = []

            # nesting: primarily for plots (to e.g. plot several variables on the same plot)
            nesting = []

            # only consider specified outputs
            dims = self.dimSpecs.copy()
            dims['output'] = r['outputs']

            dimSizes = [len(dims[d]) for d in r['order']]

            for d1 in xrange(dimSizes[0]):

                for d2 in xrange(dimSizes[1]):

                    for d3 in xrange(dimSizes[2]):

                        idx = [d1, d2, d3]
                        v = [(r['order'][i], dims[r['order'][i]][idx[i]]) \
                             for i in xrange(3)]

                        if d2 == 0 and d3 == 0:
                            nesting.append((v[0][1], []))

                        if d3 == 0:
                            nesting[d1][1].append((v[1][1], []))

                        nesting[d1][1][d2][1].append(len(rvars) + len(addVars))

                        v = dict(v)

                        v.update({'kerasName': getKerasName(v),
                                  'name': '%s%s_%d' % (v['set'], v['metric'], v['output']), 'val': []})

                        rvars.append(v)

            r['vars'] = addVars + rvars
            r['nesting'] = addNesting + nesting

    def on_epoch_end(self, epoch, logs={}):

        for r in self.reports:

            for idx, v in reversed(list(enumerate(r['vars']))):

                # save variables into the report objects

                if v['kerasName'] in logs.keys():
                    v['val'].append(logs[v['kerasName']])
                elif epoch == 0:
                    print(idx)
                    del r['vars'][idx]

            # report

            if r['type'] == 'text':
                self.writeCSV(r, logs, True if epoch == 0 else False)
            elif r['type'] == 'plot':
                self.plot(r, logs)

    def writeCSV(self, report, logs, rewrite=False):

        with open(os.path.join(self.savePaths, 'evolution.csv'), "w" if rewrite else "a") as myfile:
            writer = csv.writer(myfile, delimiter=';')

            rvars = report['vars']

            if rewrite:
                writer.writerow([rvar['name'] for rvar in rvars])

            writer.writerow([rvar['val'][-1] for rvar in rvars])

    def plot(self, report, logs):

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        lineStyle = ['dashed', 'solid', 'dotted']

        rvars = report['vars']
        nesting = report['nesting']

        plt.figure(figsize=(10, 10 * len(nesting)))
        #        plt.figure(figsize=(20, 60))
        gs = gridspec.GridSpec(len(nesting), 1, height_ratios=[1] * (len(nesting)))

        # plot
        for i1, l1 in enumerate(nesting):

            ax = plt.subplot(gs[i1])
            plt.title(l1[0])
            plt.ylabel(l1[0])

            for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
                item.set_fontsize(30)

            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)

            # color
            for i2, l2 in enumerate(l1[1]):

                # line fill
                for i3, l3 in enumerate(l2[1]):
                    rvars[l3]['val']

                    line, = plt.plot(range(0, len(rvars[l3]['val'])), rvars[l3]['val'], \
                                     ls=lineStyle[i3], color=colors[i2], label=rvars[l3]['name'])

            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

        #            ax.legend(bbox_to_anchor=(1.2, 0.5), fontsize = 20)

        try:

            plt.savefig(os.path.join(self.savePaths, report['file']))

        except Exception as inst:

            print(type(inst))
            print(inst)

        plt.close()


class LossHistoryVAE(callbacks.Callback):

    def __init__(self, savePaths, model, valid_set_x):
        super(callbacks.Callback, self).__init__()

        self.valid_set_x = valid_set_x[:3]
        self.savePath = savePaths[0]
        self.model = model
        self.original_dim = np.prod(np.shape(valid_set_x[0]))

    def on_epoch_end(self, epoch, logs={}):

        # compute losses
        rec_loss_val, kl_loss_val = self.compute_losses(self.model, self.valid_set_x)

        # save in csv
        path_overall = os.path.join(self.savePath, 'evolution2losses.csv')
        if epoch == 0:
            self.writeNamesCSV(['rec_loss', 'kl_loss'], path_overall)
        self.writeCSV([rec_loss_val * self.original_dim, kl_loss_val], path_overall)

    def compute_losses(self, model, valid_set_x):
        nbr_samples = len(valid_set_x)
        rec_loss_list = np.zeros(nbr_samples)
        kl_loss_list = np.zeros(nbr_samples)

        for i in range(nbr_samples):
            [x_decoded_mean, z_mean, z_log_var] = model.predict(np.expand_dims(valid_set_x[i], axis=0))
            rec_loss_list[i] = K.eval(self.rec_loss(valid_set_x, x_decoded_mean))
            kl_loss_list[i] = K.eval(self.kl_loss(z_log_var, z_mean))

        return np.mean(rec_loss_list), np.mean(kl_loss_list)

    def writeCSV(self, losses, pathWrite):
        with open(pathWrite, "a") as myfile:
            writer = csv.writer(myfile, delimiter=';')
            writer.writerow(losses)  # should be a list

    def writeNamesCSV(self, names, pathWrite):
        with open(pathWrite, "w") as myfile:
            writer = csv.writer(myfile, delimiter=';')
            writer.writerow(names)

    def rec_loss(self, x, x_decoded_mean):
        return K.mean(metrics.mean_squared_error(x, x_decoded_mean))

    def kl_loss(self, z_log_var, z_mean):
        return - .5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)


class LossHistoryBinary(callbacks.Callback):

    def __init__(self, savePaths, model, valid_set_x, valid_set_y):
        super(callbacks.Callback, self).__init__()

        self.valid_set_x = valid_set_x
        self.valid_set_y = valid_set_y

        self.savePath = savePaths
        self.model = model

    def on_epoch_end(self, epoch, logs={}):

        # compute losses
        bce_loss_list, acc_loss_list = self.compute_losses(self.model, self.valid_set_x, self.valid_set_y)

        # save in csv
        path_overall = os.path.join(self.savePath, 'evolution2losses.csv')
        if epoch == 0:
            self.writeNamesCSV(['bce_loss', 'acc_loss'], path_overall)
        self.writeCSV([bce_loss_list, acc_loss_list], path_overall)

    def compute_losses(self, model, valid_set_x, valid_set_y):
        nbr_samples = len(valid_set_x)
        bce_loss_list = np.zeros(nbr_samples)
        acc_loss_list = np.zeros(nbr_samples)

        for i in range(nbr_samples):
            y_pred = model.predict(np.expand_dims(valid_set_x[i], axis=0))
            x = y_pred[0]
            z = valid_set_y[i]

            bce_loss_list[i] = max(x, 0) - x * z + math.log(1 + math.exp(-abs(x)))
            acc_loss_list[i] = float(z == round(x))

        return np.mean(bce_loss_list), np.mean(acc_loss_list)

    def writeCSV(self, losses, pathWrite):
        with open(pathWrite, "a") as myfile:
            writer = csv.writer(myfile, delimiter=';')
            writer.writerow(losses)  # should be a list

    def writeNamesCSV(self, names, pathWrite):
        with open(pathWrite, "w") as myfile:
            writer = csv.writer(myfile, delimiter=';')
            writer.writerow(names)


class Reached10Epopchs(callbacks.Callback):
    def __init__(self, savePaths):
        super(callbacks.Callback, self).__init__()
        self.savePath = savePaths[0]

    def on_epoch_end(self, epoch, logs={}):
        if epoch == 10:
            with open(os.path.join(self.savePath, '10EpochsReached.csv'), "w") as myfile:
                writer = csv.writer(myfile, delimiter=';')
                writer.writerow([])


class saveEvery50Models(callbacks.Callback):
    def __init__(self, savePaths):
        super(callbacks.Callback, self).__init__()
        self.savePath = savePaths[0]

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 50 == 0 and epoch != 0:
            self.model.save_weights(os.path.join(self.savePath, 'best_weights_' + str(epoch) + '.hdf5'), overwrite=True)


class recordGradients(callbacks.Callback):
    def __init__(self, train_set_x, savePaths, model, perSample):
        super(callbacks.Callback, self).__init__()

        self.train_set_x = train_set_x
        self.savePath = savePaths[0]
        self.model = model
        self.perSample = perSample
        os.mkdir(os.path.join(self.savePath, 'gradientsPerEpoch'))

    def on_epoch_end(self, epoch, logs={}):

        meanGrad, layer_names, gradPerSample = self.compute_gradients(self.model, self.train_set_x)

        # save overall gradient
        path_overall = os.path.join(self.savePath, 'gradients.csv')
        if epoch == 0:
            self.writeNamesCSV(layer_names, path_overall)
        self.writeCSV(meanGrad, path_overall)

        # save gradients of the current epoch
        if self.perSample:
            path_epoch = os.path.join(self.savePath, 'gradientsPerEpoch', str(epoch))
            path_epoch_csv = os.path.join(path_epoch, 'gradients.csv')
            os.mkdir(path_epoch)
            self.writeNamesCSV(layer_names, path_epoch_csv)
            for i in range(len(self.train_set_x)):
                self.writeCSV(gradPerSample[i], path_epoch_csv)

    def compute_gradients(self, model, train_set_x):
        # define the function
        weights = model.trainable_weights  # weight tensors
        #        weights = [weight for weight in weights if model.get_layer(weight.name.split('/')[0]).trainable] # filter down weights tensors to only ones which are trainable
        gradients = model.optimizer.get_gradients(model.total_loss, weights)  # gradient tensors

        input_tensors = [model.inputs[0],  # input data
                         model.sample_weights[0],  # how much to weight each sample by
                         model.targets[0],  # labels
                         K.learning_phase(),  # train or test mode
                         ]

        get_gradients = K.function(inputs=input_tensors, outputs=gradients)

        # run on the whole epoch and average
        nbr_layers = len(weights)
        meanGrad = np.zeros(nbr_layers)
        gradPerSample = np.zeros([len(train_set_x), nbr_layers])
        for j, image in enumerate(train_set_x):
            inputs = [np.expand_dims(image, axis=0),  # X
                      [1],  # sample weights
                      [[1]],  # y
                      0  # learning phase in TEST mode
                      ]

            grad = get_gradients(inputs)

            # average gradients per layer
            for i, g in enumerate(grad):
                meanGrad[i] += np.mean(g)
                gradPerSample[j, i] = np.mean(g)

        meanGrad = meanGrad * 1. / len(train_set_x)
        layer_names = [weight.name.split('/')[0] for weight in weights]
        return meanGrad.tolist(), layer_names, gradPerSample

    def writeCSV(self, gradients, pathWrite):
        with open(pathWrite, "a") as myfile:
            writer = csv.writer(myfile, delimiter=';')
            writer.writerow(gradients)  # gradient should be a list:  dimensions account for layers

    def writeNamesCSV(self, names, pathWrite):
        with open(pathWrite, "w") as myfile:
            writer = csv.writer(myfile, delimiter=';')
            writer.writerow(names)




