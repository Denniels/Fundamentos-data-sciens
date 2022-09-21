#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: lec6_graphs.py
Author: Ignacio Soto Zamorano
Email: ignacio[dot]soto[dot]z[at]gmail[dot]com
Github: https://github.com/ignaciosotoz
Description: Ancilliary files for categorical classification methods - Fundamentos Data Science - ADL
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from sklearn.metrics import roc_curve

def inverse_logit(x):
    """docstring for inverse_logit"""
    return 1/(1+np.exp(-x))

def residual_deviance(df, model, estimates=[]):
    """docstring for residual_deviance"""
    X_mat = df.loc[:, estimates]
    estimate_beta = model.params
    beta_prob = inverse_logit(np.matmul(X_mat, estimate_beta))
    log_score = np.sum(df[model.params.index[0]] * np.log(beta_prob) + (1 - df[model.params.index[0]]) * np.log(1 - beta_prob)) 
    return -2 * log_score

def null_deviance(df, model, estimates=[]):
    """docstring for null_deviance"""
    target_mean = np.mean(df[model.params.index[0]])
    return -2 * len(df) * np.log(target_mean) + len(df) * (1 - target_mean) * np.log(1 - target_mean)

# def crossvalidation_schema(cv_folds=10,ax, textprop={}):
    # """docstring for crossvalidation_schema"""
    # for i in range(cv_folds):
        # ax.add_patch(plt.Rectangle((0, i), 5, 0.7, fc='lightgrey'))
        # ax.add_patch(plt.Rectangle((5. * i /cv_folds, i), 5. / cv_folds, 0.7, fc='skyblue'))
        # ax.text(5. * (i + .5) / cv_folds, i + .35, "Validación", ha='center', va='center', **textprop)
        # ax.text(5, i + .35, r'Ensayo {0}$\in${1}    '.format(cv_folds - i, cv_folds), ha = 'right', va='center', **textprop)
        # ax.set_xlim(-1, 6)
        # ax.set_ylim(-0.2, cv_folds + 0.2)

def logit_probit_lpm():
    """docstring for logit_probit_lpm"""
    x_axis = np.linspace(-6, 6, 100)
    y_axis = np.linspace(0, 1, 100)
    plt.plot(x_axis, stats.logistic.cdf(x_axis), label = "Logistic", color='tomato', lw=3)
    plt.plot(x_axis, stats.norm.cdf(x_axis), label = "Probit", color='dodgerblue', lw=3)
    plt.plot(x_axis, y_axis, label = "LPM", color='darkgoldenrod', lw=3)
    plt.axvline(x=0, ymin=0.0, ymax=0.5, linestyle='--', lw=1, color='grey')
    plt.axhline(y=.5, xmin=0.0, xmax=.5, linestyle='--', lw=1, color='grey')
    plt.axvline(x=2, ymin=0.0, ymax=.94, linestyle='--', lw=1, color='grey')
    plt.axhline(y=.67, xmin=0.0, xmax=.65, linestyle='--', lw=1, color='darkgoldenrod')
    plt.axhline(y=.98, xmin=0.0, xmax=.65, linestyle='--', lw=1, color='dodgerblue' )
    plt.axhline(y=.88, xmin=0.0, xmax=.65, linestyle='--', lw=1, color='tomato')
    plt.text(-6, .68, r'LPM: $\hat{p}$=.68', color='darkgoldenrod')
    plt.text(-6, .89, r'Logistic: $\hat{p}$=.88', color='tomato')
    plt.text(-6, .99, r'Probit: $\hat{p}$=.98', color='dodgerblue')
    plt.text(-6, .51, r'LPM, Logistic, Probit: $\hat{p}$=.5')
    plt.legend()

def concise_summary(mod, print_fit=True):
    """docstring for concise_summary"""
    fit = pd.DataFrame({'Statistics': mod.summary2().tables[0][2][2:], 'Value': mod.summary2()[0][3][2:]})
    estimates = pd.DataFrame(mod.summary2().tables[1].loc[:, 'Coef.':'Std.Err.'])
    if print_fit is True:
        print("\n\nGoodness of Fit statistics\n", fit)
        print("\n\nPoint estimates\n\n", estimates)

def summary_nas(df):
    """docstring for summary_nas"""
    varname = []
    na = []
    na_perc = []

    for i in df.columns:
        retrieve_nas = df[i].isna().value_counts()
        varname.append(i)
        na.append(retrieve_nas[1])
        na_perc.append(retrieve_nas[1] / sum(retrieve_nas))

    features = pd.DataFrame({'variable': varname, 
                             'total_na': na,
                             'na_percentage': na_perc})
    return features

def plot_roc(test_vector, true_vector):
    """docstring for plot_roc"""
    false_positive, true_positive, threshold = roc_curve(test_vector, true_vector)
    plt.plot(false_positive, true_positive)
    plt.plot([0, 1], linestyle='--')
    plt.plot([0, 0], [1, 0])
    plt.plot([1, 1])
    plt.title("ROC Curve")
    plt.ylabel('True Positive')
    plt.xlabel('False Positive')

def recall(conf_mat):
    true_positive = conf_mat[0][0]
    false_positive = conf_mat[0][1]
    false_negative = conf_mat[1][0]
    true_negative = conf_mat[1][1]
    """docstring for recall"""
    return true_positive / (true_positive + false_negative) 

def precision(conf_mat):
    """docstring for precision"""
    true_positive = conf_mat[0][0]
    false_positive = conf_mat[0][1]
    false_negative = conf_mat[1][0]
    true_negative = conf_mat[1][1]
    """docstring for precision"""
    return true_positive / (true_positive + false_positive)

def accuracy(conf_mat):
    """docstring for accuracy"""
    true_positive = conf_mat[0][0]
    false_positive = conf_mat[0][1]
    false_negative = conf_mat[1][0]
    true_negative = conf_mat[1][1]
    correct = true_positive + true_negative
    total = true_positive + true_negative + false_positive + false_negative
    return correct / total

def f1_score(conf_mat):
    """docstring for f1_score"""
    pres = precision(conf_mat)
    rec = recall(conf_mat)
    return 2 * pres * rec / (pres + rec)

def logit_probit_lpm_redo(x_value=0,n_size=100, x_axis=np.linspace(-6,6,100)):
    """docstring for logit_probit_lpm_redo"""

    # plot Probability space for lpm
    y_axis = np.linspace(0, 1, n_size)
    # aproximate
    lpm_slope = 1/int(x_axis.max() - x_axis.min())

    # plot curves
    plt.plot(x_axis, stats.logistic.cdf(x_axis), label = 'Logistic', color = 'tomato')
    plt.plot(x_axis, stats.norm.cdf(x_axis), label = "Probit", color='dodgerblue', lw=3)
    plt.plot(x_axis, y_axis, label = "LPM", color='darkgoldenrod', lw=3)

    # store estimates
    store_estimates = {'Logit': stats.logistic.ppf(x_value),
                       'Probit': stats.norm.cdf(x_value),
                       'LPM': lpm_slope * x_value}
    colors = ['tomato', 'dodgerblue', 'darkgoldenrod']

    # loop thru dict
    for key, value in store_estimates.items():
        plt.axvline(x=x_value, ymin=0, ymax=value, label=key)
        plt.text(x_value, value, r'{0}:Pr={1}'.format(key, round(value, 2)))
        plt.legend()
