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
START_ID=2400
series_exp = [range(START_ID+1,START_ID+10),range(START_ID+11,START_ID+20), range(START_ID+21,START_ID+30),range(START_ID+31,START_ID+40),range(START_ID+41,START_ID+50),range(START_ID+51,START_ID+60),range(START_ID+61,START_ID+70),range(START_ID+71,START_ID+80),range(START_ID+81,START_ID+90),range(START_ID+91,START_ID+100)]

# paths
path_experiments = '../../experiments'
path_save = os.path.join(path_experiments, EXPERIMENT_ID)

# create exp folder
createExpFolderandCodeList(path_save)

# iterate over metrics
for metric in METRICS:
    average_df = pd.DataFrame()
    for series in series_exp:
        average_dict = {}
        for risk_level, exp_id in enumerate(series):
            df = pd.read_csv(os.path.join(path_experiments, str(exp_id), 'metrics.csv'))
            average_dict[str(risk_level)] = df[metric][0]
        # update Dataframe
        average_df = average_df.append(average_dict, ignore_index=True)

    # compute average and std of averages
    mean = average_df.mean()
    std = average_df.std()

    # plot figure
    plt.figure()
    plt.plot(range(1,len(mean)+1),mean)
    plt.fill_between(range(1,len(mean)+1), mean-std, mean+std, color = 'blue', alpha = 0.15)
    plt.xlabel('Risk Level')
    plt.ylabel(metric.capitalize())
    plt.savefig(os.path.join(path_save, metric+'.pdf'))
    plt.close()

    # save res in csv
    average_df.loc['mean'] = mean
    average_df.loc['std'] = std
    average_df.to_csv(os.path.join(path_save,metric+'.csv'))
