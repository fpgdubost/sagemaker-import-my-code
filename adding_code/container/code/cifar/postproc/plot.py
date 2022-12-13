import matplotlib

matplotlib.use('Agg')
from matplotlib import gridspec
import matplotlib.pyplot as plt
import pandas as pd
from ipdb import set_trace as bp
import sys, os
from basic_functions import createExpFolderandCodeList

EXPERIMENT_ID = sys.argv[1]
METRICS = ['f1','acc','ap','auc','precis','recall']
series_exp = range(111,120)

# paths
path_experiments = '../../experiments'
path_save = os.path.join(path_experiments, EXPERIMENT_ID)

# create exp folder
createExpFolderandCodeList(path_save)

# iterate over metrics
for metric in METRICS:
    list_average = []
    list_lb = []
    list_ub = []
    for exp_id in series_exp:
        df = pd.read_csv(os.path.join(path_experiments, str(exp_id), 'metrics.csv'))
        list_average.append(df[metric][0])
        list_lb.append(df[metric][1])
        list_ub.append(df[metric][2])

    # plot figure
    plt.figure()
    plt.plot(range(1, len(list_average) + 1), list_average)
    plt.fill_between(range(1, len(list_average) + 1), list_lb, list_ub, color='blue', alpha=0.15)
    plt.savefig(os.path.join(path_save, metric + '.pdf'))
    plt.xlabel('Risk Level')
    plt.xlabel(metric)
    plt.close()
