import matplotlib

matplotlib.use('Agg')
from matplotlib import gridspec
import matplotlib.pyplot as plt
import pandas as pd
from ipdb import set_trace as bp
import sys, os
from basic_functions import createExpFolderandCodeList
from scipy.stats import ttest_ind


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


def boostrapping_hypothesisTesting(data_method1, data_method2, nbr_runs=100000):
    n = len(data_method1.index)
    m = len(data_method2.index)
    total = n + m

    # compute the metric for both method
    metric_method1 = custom_metric(data_method1)
    metric_method2 = custom_metric(data_method2)

    # compute statistic t
    t = abs(metric_method1 - metric_method2)

    # merge data from both methods
    data = pd.concat([data_method1, data_method2])

    # compute bootstrap statistic
    nbr_cases_higher = 0
    for r in range(nbr_runs):
        # sample random indexes with replacement
        ind = np.random.randint(total, size=total)

        # select random subset with replacement
        data_bootstrapped = data.iloc[ind]

        # split into two groups
        data_bootstrapped_x = data_bootstrapped[:n]
        data_bootstrapped_y = data_bootstrapped[n:]

        # compute metric for both groups
        metric_x = custom_metric(data_bootstrapped_x)
        metric_y = custom_metric(data_bootstrapped_y)

        # compute bootstrap statistic
        t_boot = abs(metric_x - metric_y)

        # compare statistics
        if t_boot > t:
            nbr_cases_higher += 1

    pvalue = nbr_cases_higher * 1. / nbr_runs
    print(nbr_cases_higher)
    print(pvalue)

    return pvalue

if __name__ == '__main__':

    EXPERIMENT_ID = sys.argv[1]
    METRICS = ['f1','acc','ap','auc','precis','recall']
    ALL_RES_ID=[800,1100,1300,1500,1700,1900,2100,2300,2500]

    # paths
    path_experiments = '../../experiments'
    path_save = os.path.join(path_experiments, EXPERIMENT_ID)

    # create exp folder
    createExpFolderandCodeList(path_save)

    # iterate over metrics
    for metric in METRICS:
        df_res = pd.DataFrame()
        df_stats = pd.DataFrame()
        for res_id in ALL_RES_ID:
            # read results
            df = pd.read_csv(os.path.join(path_experiments,str(res_id),metric+'.csv'))
            # compute stat test
            rep_10_exp_no_risk = df['0'][:-2]
            stat_dict = {}
            for risk_level in df.columns[2:]:

                res_10_rep_curr_risk = df[risk_level][:-2]
                stat_dict[risk_level] = '{:.3f}'.format(ttest_ind(rep_10_exp_no_risk,res_10_rep_curr_risk,nan_policy='omit')[1])
            # add std as as str
            res_dict = df.iloc[10]
            for risk_level in res_dict.index[1:]:
                idx = int(risk_level) + 1
                res_dict[idx] = '{:.2f}'.format(res_dict[idx]) + ' +/- ' + '{:.2f}'.format(df.iloc[11][idx])
            # store results
            df_res = df_res.append(res_dict, ignore_index=True)
            df_stats = df_stats.append(stat_dict,ignore_index=True)

        # save results to disk
        df_res.to_csv(os.path.join(path_save,metric+'.csv'))
        df_stats.to_csv(os.path.join(path_save,metric+'_p_values.csv'))
