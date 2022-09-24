#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
File: lec7_graphs.py
Author: Ignacio Soto Zamorano
Email: ignacio[dot]soto[dot]z[at]gmail[dot]com
Github: https://github.com/ignaciosotoz
Description: Ancilliary files for dimensionality reduction methods - Fundamentos Data Science - ADL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


def pca_variance():
    """
    Figure form Hands On Machine Learning -  Aurelien Géron
    """
    angle = np.pi / 5
    stretch = 5
    m = 200

    np.random.seed(3)
    X = np.random.randn(m, 2) / 10
    X = X.dot(np.array([[stretch, 0],[0, 1]])) # stretch
    X = X.dot([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]) # rotate

    u1 = np.array([np.cos(angle), np.sin(angle)])
    u2 = np.array([np.cos(angle - 2 * np.pi/6), np.sin(angle - 2 * np.pi/6)])
    u3 = np.array([np.cos(angle - np.pi/2), np.sin(angle - np.pi/2)])

    X_proj1 = X.dot(u1.reshape(-1, 1))
    X_proj2 = X.dot(u2.reshape(-1, 1))
    X_proj3 = X.dot(u3.reshape(-1, 1))

    plt.figure(figsize=(8,4))
    plt.subplot2grid((3,2), (0, 0), rowspan=3)
    plt.plot([-1.4, 1.4], [-1.4*u1[1]/u1[0], 1.4*u1[1]/u1[0]], color='tomato',linestyle="-", linewidth=2)
    plt.plot([-1.4, 1.4], [-1.4*u2[1]/u2[0], 1.4*u2[1]/u2[0]], color='dodgerblue', linestyle="--", linewidth=2)
    plt.plot([-1.4, 1.4], [-1.4*u3[1]/u3[0], 1.4*u3[1]/u3[0]], color='darkgoldenrod', linestyle=":", linewidth=2)
    plt.plot(X[:, 0], X[:, 1], 'o', color='grey', alpha=0.5)
    plt.axis([-1.4, 1.4, -1.4, 1.4])
    plt.arrow(0, 0, u1[0], u1[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    plt.arrow(0, 0, u3[0], u3[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    plt.text(u1[0] + 0.1, u1[1] - 0.05, r"$\mathbf{c_1}$", fontsize=22)
    plt.text(u3[0] + 0.1, u3[1], r"$\mathbf{c_2}$", fontsize=22)
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$x_2$", fontsize=18, rotation=0)

    plt.subplot2grid((3,2), (0, 1))
    plt.plot([-2, 2], [0, 0], "k-", linewidth=1)
    plt.plot(X_proj1[:, 0], np.zeros(m), 'o', color='tomato', alpha=0.3)
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().get_xaxis().set_ticklabels([])
    plt.axis([-2, 2, -1, 1])

    plt.subplot2grid((3,2), (1, 1))
    plt.plot([-2, 2], [0, 0], "k--", linewidth=1)
    plt.plot(X_proj2[:, 0], np.zeros(m), 'o', color='dodgerblue', alpha=0.3)
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().get_xaxis().set_ticklabels([])
    plt.axis([-2, 2, -1, 1])

    plt.subplot2grid((3,2), (2, 1))
    plt.plot([-2, 2], [0, 0], "k:", linewidth=2)
    plt.plot(X_proj3[:, 0], np.zeros(m), 'o', color='darkgoldenrod', alpha=0.3)
    plt.gca().get_yaxis().set_ticks([])
    plt.axis([-2, 2, -1, 1])
    plt.xlabel("$z_1$", fontsize=18)



# Simulate distance in d-dimensional space

def sim_distance(n_dims=2):
    """
    figure from Data Science Handbook - Fieldy Cady
    """
    sim_data = np.random.uniform(size=n_dims * 1000).reshape((1000, n_dims))
    fetch_data = spatial.distance.cdist(sim_data, sim_data)
    tmp = pd.Series(fetch_data.reshape(1000000))
    plt.hist(tmp, bins=50, ec='white', color='slategrey')
    plt.axvline(np.mean(tmp), color='tomato', lw=4, label='Distancia Promedio')
    plt.title("D = {}".format(n_dims))




bivariate_mu_params = np.array([[ 0.2,  2.3],[-1.5 ,  2.3],[-2.8,  1.8],[-1.8,  2.8], [-.8,  1.3]])
bivariate_sigma_params = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

X, y = make_blobs(n_samples=2000, centers=bivariate_mu_params,
                  cluster_std=bivariate_sigma_params, random_state=7)

def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)

def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                 cmap="viridis", alpha=.5)
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom='off')
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')

def plot_pca_components(x, coefs=None, mean=0, cmps=None,
                        imshape=(8, 8), n_components=10, fontsize=12,
                        show_mean=True):

    """
    figure from Data Science with Python - Jake Vanderplas
    """


    if coefs is None:
        coefs = x

    if cmps is None:
        cmps = np.eye(len(coefs), len(x))

    mean = np.zeros_like(x) + mean


    fig = plt.figure(figsize=(1.2 * (5 + n_components), 1.2 * 2))
    grid = plt.GridSpec(2, 4 + bool(show_mean) + n_components, hspace=0.3)

    def show(i, j, x, title=None):
        ax = fig.add_subplot(grid[i, j], xticks=[], yticks=[])
        ax.imshow(x.reshape(imshape), interpolation='nearest', cmap="Blues")
        if title:
            ax.set_title(title, fontsize=fontsize)

    show(slice(2), slice(2), x, "Verdadero")

    approx = mean.copy()

    counter = 2
    if show_mean:
        show(0, 2, np.zeros_like(x) + mean, r'$\mu$')
        show(1, 2, approx, r'$1 \cdot \mu$')
        counter += 1

    for i in range(n_components):
        approx = approx + coefs[i] * cmps[i]
        show(0, i + counter, cmps[i], r'$c_{0}$'.format(i + 1))
        show(1, i + counter, approx, r"${0:.2f} \cdot c_{1}$".format(coefs[i], i + 1))
        if show_mean or i > 0:
            plt.gca().text(0, 1.05, '$+$', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=fontsize)

    show(slice(2), slice(-2, None), approx, "Aproximación")
    return fig
