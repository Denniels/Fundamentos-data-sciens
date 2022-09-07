#!/usr/bin/env python
# -*- coding: utf-8 -*-


import random
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as plt
import statsmodels.formula.api as smf

def coefplot(model, varnames=True, intercept=False, fit_stats=True, figsize=(7, 12)):
    """
    coefplot - Visualize coefficient magnitude and approximate frequentist significance from a model.

    @parameters:
    - model: a `statsmodels.formula.api` class generated method, which must be already fitted.
    - varnames: if True, y axis will contain the name of all of the variables included in the model. Default: True
    - intercept: if True, coefplot will include the $\beta_{0}$ estimate. Default: False.
    - fit_stats: if True, coefplot will include goodness-of-fit statistics. Default: True.

    @returns:
    - A `matplotlib` object.
    """
    if intercept is True:
        coefficients = model.params.values
        errors = model.bse
        if varnames is True:
            varnames = model.params.index
        else:
            coefficients = model.params.values[1:]
        errors = model.bse[1:]
        if varnames is True:
            varnames = model.params.index[1:]

    tmp_coefs_df = pd.DataFrame({'varnames': varnames, 'coefs': coefficients,'error_bars': errors})
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(y=tmp_coefs_df['varnames'], x=tmp_coefs_df['coefs'],
                xerr=tmp_coefs_df['error_bars'], fmt='o',
                color='slategray', label='Estimated point')
    ax.axvline(0, color='tomato', linestyle='--', label='Null Effect')
    ax.set_xlabel(r'$\hat{\beta}$')
    fig.tight_layout()
    plt.legend(loc='best')

    if fit_stats is True:
        if 'linear_model' in model.__module__.split('.'):
            plt.title(r'R$^{2}$' + "={0}, f-value={1}, n={2}".format(round(model.rsquared, 2),
                                                                     round(model.f_pvalue, 3),
                                                                     model.nobs))
        elif 'discrete_model' in model.__module__.split('.'):
            plt.title("Loglikelihood = {0}, p(ll-Rest)={1}, n={2}".format(round(model.llf, 2),
                                                                          round(model.llr_pvalue, 3),
                                                                          model.nobs))
