#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
File: lec5_graphs.py
Author: Ignacio Soto Zamorano
Email: ignacio[dot]soto[dot]z[at]gmail[dot]com
Github: https://github.com/ignaciosotoz
Description: Ancilliary file for intro to data science - adl
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from sklearn.preprocessing import PolynomialFeatures as polynom
from sklearn.linear_model import LinearRegression as linreg
from sklearn.pipeline import make_pipeline
plt.style.use('seaborn-whitegrid')

def feature_target():
    """Plot feature and target illustration"""
    figure = plt.figure()
    ax = figure.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.axis('equal')

    ax.vlines(range(6), ymin=0, ymax=9, lw=2, color='dodgerblue')
    ax.hlines(range(10), xmin=0, xmax=5, lw=2, color='dodgerblue')
    font_properties = dict(size=10, family='monospace')
    ax.text(1, -1, r'Matriz\b Atributos ($X$)', size=14)
    ax.text(0.1, -0.3, r'cantidad (p)$\longrightarrow$', **font_properties)
    ax.text(-0.1, 0.1, r'$\longleftarrow$ tamaño_muestral', rotation=90, va='top', ha='right', **font_properties)

    ax.vlines(range(8, 10), ymin=0, ymax=9, lw=2, color='tomato')
    ax.hlines(range(10), xmin=8, xmax=9, lw=2, color='tomato')
    ax.text(7, -1, r'Vector\b Objetivo ($y$)', size=14)
    ax.text(7.9, 0.1, r'$\longleftarrow$ tamaño_muestral', rotation=90, va='top', ha='right', **font_properties)

    ax.set_ylim(10, -2)

def crossvalidation(cv_folds, ax, text_properties={}):
    """Plot crossvalidation illustration"""
    for i in cv_folds:
        ax.add_patch(plt.Rectangle((0, i), 5, 0.7, fc='lightgrey'))
        ax.add_patch(plt.Rectangle((5. * i / cv_folds, i), 5. /cv_folds, 0.7, fc='skyblue'))
        ax.text(5. * (i + .5)/ cv_folds, i + .35, "Validación", ha='center', va='center', **text_properties)
        ax.text(0, i + .35, r'Ensayo {0}$\in${1}    '.format(cv_folds - i, cv_folds), ha='right', va='center', **text_properties)
        ax.text(5, i + .35, r'$f(\hat\theta - \Theta)\longrightarrow\varepsilon$', ha='left', **text_properties)
    ax.set_xlim(-1, 6)
    ax.set_ylim(-0.2, cv_folds + 0.2)

def simulate_data(N=40, epsilon=1.0, pseudorandom=11238):
    """Simulate fictional data"""
    random_seed = np.random.RandomState(pseudorandom)
    X_mat = random_seed.rand(N, 1) ** 2
    y_vec = 10 -1 / (X_mat.ravel() + 0.1)
    if epsilon > 0:
        y_vec +=  epsilon + random_seed.randn(N)
    return X_mat, y_vec

def polynomial_reg(degree=2, **kwargs):
    """Pipeline regression models"""
    return make_pipeline(polynom(degree), linreg(**kwargs))

def bias_variance():
    """Plot bias variance tradeoff illustration"""
    X_mat, y_vec = simulate_data()
    X_mat_fit = np.linspace(-0.1, 1.0, 1000)[:, None]
    mod_1 = polynomial_reg(1).fit(X_mat, y_vec)
    mod_7 = polynomial_reg(7).fit(X_mat, y_vec)
    mod_15 = polynomial_reg(15).fit(X_mat, y_vec)

    fig, ax = plt.subplots(1, 3)

    ax[0].scatter(X_mat.ravel(), y_vec, s=40)
    ax[0].plot(X_mat_fit.ravel(), mod_1.predict(X_mat_fit), color='tomato')
    ax[0].axis([-0.1, 1.0, -2, 12])
    ax[0].set_title('Sesgo alto')
    ax[1].scatter(X_mat.ravel(), y_vec, s=40)
    ax[1].plot(X_mat_fit.ravel(), mod_7.predict(X_mat_fit), color='tomato')
    ax[2].scatter(X_mat.ravel(), y_vec, s=40)
    ax[1].axis([-0.1, 1.0, -2, 12])
    ax[1].set_title('Caso intermedio')
    ax[2].plot(X_mat_fit.ravel(), mod_15.predict(X_mat_fit), color='tomato')
    ax[2].axis([-0.1, 1.0, -2, 12])
    ax[2].set_title('Varianza Alta')

def validate_curve():
    xaxis = np.linspace(0, 1, 1000)
    y_1 = -(xaxis - .5) ** 2
    y_2 = y_1 - .33 + np.exp(xaxis - 1)
    plt.plot(xaxis, y_2, lw=8, color='tomato')
    plt.text(0.42, .35, "Entrenado", color='tomato', size=14)
    plt.plot(xaxis, y_1, lw=8, color='dodgerblue')
    plt.text(0.71, -.03, "Validado", color='dodgerblue', size=14)
    plt.xlim(0, 1)
    plt.fill_between(xaxis, -.6, .6, facecolor='grey', alpha=.35, where=xaxis < .2)
    plt.text(0.05, 0.45, "Sesgo\nAlto", size=14)
    plt.fill_between(xaxis, -.6, .6, facecolor='grey', alpha=.35, where=xaxis > .8)
    plt.text(0.85, .47, "Varianza\nAlta", size=14)
    plt.text(0.45, 0.05, "Mejor Modelo", color='dodgerblue')
    plt.ylabel(r'Puntaje $\rightarrow$')
    plt.xlabel(r'Complejidad del modelo $\rightarrow$')
    plt.xticks([])
    plt.yticks([])

def learning_curve():
    xaxis = np.linspace(0, 1, 1000)
    y_1 = 0.75 + 0.2 * np.exp(-4 * xaxis)
    y_2 = 0.7 - 0.6 * np.exp(-4 * xaxis)

    plt.plot(xaxis,y_1, lw=8, color='tomato')
    plt.text(0.10,0.95, "Entrenado", color='tomato',size=14)
    plt.plot(xaxis, y_2, lw=8, color='dodgerblue')
    plt.text(0.15, 0.25, "Validado", color='dodgerblue', size=14)
    plt.fill_between(xaxis, -.6, 1.2, facecolor='grey', alpha=.35, where=xaxis < .2)
    plt.text(0.02, 0.45, "Varianza\nAlta", size=14)
    plt.fill_between(xaxis, -.6, 1.2, facecolor='grey', alpha=.35, where=xaxis > .8)
    plt.text(0.85, .30, "Mejor\nAjuste", size=14)
    plt.xlim(0, 1)
    plt.ylabel(r'Puntaje $\rightarrow$')
    plt.xlabel(r' Tamaño training set $\rightarrow$')
    plt.xticks([])
    plt.yticks([])

def draw_rects(cv_folds, ax, text_properties={}):
    """
    Draw crossvalidation rectangles
    """
    for i in range(cv_folds):
        ax.add_patch(plt.Rectangle((0, i), 5, 0.7, fc='lightgrey'))
        ax.add_patch(plt.Rectangle((5. * i / cv_folds, i), 5. / cv_folds, 0.7, fc='skyblue'))
        ax.text(5. * (i + 0.5) / cv_folds, i + 0.35,
                "Validación", ha='center', va='center', **text_properties)
        ax.text(0, i + 0.35, r'Ensayo {0}$\in${1}'.format(cv_folds - i, cv_folds),
                ha='right', va='center', **text_properties)
        ax.text(5, i+.35, r'$f(\hat\theta - \Theta)\longrightarrow\varepsilon_{{}}$', ha='left', **text_properties)
 
    ax.set_xlim(-1, 6)
    ax.set_ylim(-0.2, cv_folds + 0.2)

def train_testing():
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.axis('equal')

    # Draw features matrix
    ax.vlines(range(6), ymin=0, ymax=9, lw=2, color='dodgerblue')
    font_prop = dict(size=12, family='monospace')
    ax.text(-0.1, 0.1, 'testing_features', rotation=90, va='top', ha='right', **font_prop)
    ax.text(-0.1, 10, 'training_features', rotation=90, va='top', ha='right', **font_prop)
    ax.vlines(range(6), ymin=10, ymax=19, lw=2, color='tomato')
    ax.hlines(range(20), xmin=0, xmax=5, lw=2, color='tomato')
    ax.hlines(range(10), xmin=0, xmax=5, lw=2, color='dodgerblue')

    # Draw labels vector
    ax.vlines(range(8, 10), ymin=0, ymax=9, lw=2, color='dodgerblue')
    ax.text(7.9, 0.1, 'testing_target', rotation=90, va='top', ha='right', **font_prop)
    ax.text(7.9, 10, 'training_target', rotation=90, va='top', ha='right', **font_prop)
    ax.vlines(range(8, 10), ymin=10, ymax=19, lw=2, color='tomato')
    ax.hlines(range(20), xmin=8, xmax=9, lw=2, color='tomato')
    ax.hlines(range(10), xmin=8, xmax=9, lw=2, color='dodgerblue')
    ax.set_ylim(20, -2)

def gauss_markov():
    """
    Draw \beta_{k}\sim\mathcal{N}(\cdot) approx.
    """

    xaxis = np.linspace(-3, 3, 100)
    sim_1 = stats.norm.pdf(xaxis, 0, 1)
    sim_2 = stats.norm.pdf(xaxis, 1, 1)
    sim_3 = stats.norm.pdf(xaxis, 0, .5)
    plt.plot(xaxis, sim_1, lw=3, color='dodgerblue')
    plt.annotate(r'Estimación $\hat{\theta_{1}}\neq MCO$',
                 xy=(-1.9, .35), color='dodgerblue')
    plt.plot(xaxis, sim_2, lw=3, color='darkgoldenrod')
    plt.annotate(r'Estimación $\hat{\theta_{2}}\neq MCO$',
                 xy=(1.5, .4), color='darkgoldenrod')
    plt.plot(xaxis, sim_3, lw=3, color='forestgreen')
    plt.annotate(r'Estimación $\hat{\theta_{3}}\equiv MCO$',
                 xy=(.4, .7), color='forestgreen')
    plt.vlines(0, 0, 1, color='tomato', lw=3, linestyle='--')
    plt.annotate(r'$\theta$', xy=(0.1, .9), color='tomato', fontsize=18)
    plt.title("Visualización de estimadores")

def multiple_knots():
    """
    Illustrate bias-variance tradeoff.
    """
    X_mat, y_vec = simulate_data(100)
    X_test = np.linspace(-0.1, 1.1, 500)[:, None]
    plt.scatter(X_mat.ravel(), y_vec, color='lightgrey')
    axis = plt.axis()

    # Loop for each knot on polynomial fit
    for i, deg in enumerate([1, 3, 5, 7]):
        y_test = polynomial_reg(deg).fit(X_mat, y_vec).predict(X_test)
        plt.plot(X_test.ravel(), y_test, label='deg:{0}'.format(deg))
    plt.legend()


