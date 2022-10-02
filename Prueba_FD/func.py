#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: lec4_graphs.py
Author: Ignacio Soto Zamorano
Email: ignacio[dot]soto[dot]z[at]gmail[dot]com
Github: https://github.com/ignaciosotoz
Description: Ancilliary file for intro to data science - adl
"""

import argparse
import time
import os
from collections import Counter
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy.stats import norm
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings(action="ignore")
colors = ["tomato", "darkgoldenrod", "limegreen", "dodgerblue", "sienna", "slategray"]

def generate_corr_matrix(rho_params = np.linspace(-1.0, 1.0, 20).round(1)):
    """docstring for generate_corr_matrix"""
    for i, corr in enumerate(rho_params):
        plt.subplot(4, 5, i+1)
        np.random.seed(666)
        x, y = np.random.multivariate_normal(mean = [0, 0], cov = [[1, corr], [corr, 1]], size = 100).T
        beta_1, beta_0 = np.polyfit(x, y, 1)
        plt.plot(x, y, 'o', alpha=.5)
        plt.plot(x, [beta_1 * j + beta_0 for j in x], 'b', color='tomato')
        plt.axis('off')
        plt.title(r'$\rho$={}'.format(corr), fontweight='bold')

def law_large_numbers(function = np.random.poisson, sample_size = 2000, Theta = 10):
    np.random.seed(2)
    for i in range(len(colors)):
        sample = function(Theta, size=sample_size)
        x_span = range(1, sample_size, 100)
        sample_average = [sample[:j].mean() for j in x_span]
        plt.plot(x_span, sample_average, lw=1.5, label = r'$\hat\theta$ en Ensayo {}'.format(i+1),
                color = colors[i], linestyle='--')

    plt.title('Medias muestrales y tamaño muestral')
    plt.ylabel('Media muestral')
    plt.xlabel('Tamaño muestral')
    plt.axhline(Theta, lw=3)
    plt.annotate(r'$\Theta$', xy = (sample_size - 100, 10.2), fontsize =20, color='#1c6cab')
    plt.legend()

def fdp_normal(x, mu = 0, sigma =1):
    bracket_exponencial = np.exp(-(x - mu) ** 2/ (2 * sigma ** 2))
    frac = np.sqrt(2 * np.pi) * sigma
    return (frac ** -1) * bracket_exponencial

def fdc_normal(x, mu=0, sigma=1):
    elemental = 1 + math.erf((x - mu) / np.sqrt(2) / sigma)
    return elemental / 2

def bernoulli(p):
    # genera 1 si el número aleatorio es mayor a p
    return 1 if np.random.random() < p else 0

def binomial(n, p):
    # genera la suma de ensayos de bernoulli con p probabilidad repetido n veces
    return sum(bernoulli(p) for _ in range(n))

def plot_hist (df, var):
    df[var].hist()
    
    plt.axvline(df[var].mean(), label = 'Media', color = 'orange')
    plt.axvline(np.median(df[var]), label = 'Mediana', color = 'green')

    plt.legend()
    plt.title(var)
    plt.show()

def plot_hist_one(p, n, points):
    # genera un array temporal para guardar distribuciones binomiales repetidas `points` veces
    tmp = [binomial(n, p) for _ in range(points)]
    # contador de instancias
    hist = Counter(tmp)
    # delimitador de ancho de columnas en histograma
    bins = [x -0.4 for x in hist.keys()]
    # estimador de densidad
    density = [v / points for v in hist.values()]
    plt.bar(bins, density, color='dodgerblue',alpha=.5)

    # guardamos la media y la desviación estandar
    mu = p * n
    sigma = np.sqrt(mu * (1 - p))

    # declaramos un eje x con un rango igual a la cantidad de elementos en `tmp`
    xaxis = range(min(tmp), max(tmp) + 1)
    # generar la distribución normal a partir de intervalos inferiores y superiores
    yaxis = [fdc_normal(i + 0.5, mu, sigma) - fdc_normal(i - 0.5, mu, sigma) for i in xaxis]
    # graficar
    plt.plot(xaxis, yaxis, color='tomato')
    # señalar la media
    plt.axvline(mu, color='#1c6cab', lw=2, linestyle='--')
    plt.title("Iteración: " + str(points), fontsize=10)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

def central_limit_theorem(values = [1, 5, 10, 100, 1000, 10000]):
    for i, v in enumerate(values):
        plt.subplot(2, 3, i+1)
        plot_hist(.2, 1000, v)

def confidence_intervals():
    sims = 100
    coverage = np.empty(sims) # 1 si contiene theta; 0 de lo contrario
    lower_bound = np.empty(sims) # intervalo inferior
    upper_bound = np.empty(sims) # intervalo superior

    # por cada elemento entre 1 y 100
    for i in range(sims):
        # generemos una distribución X˜N(0,1)
        draws = np.random.normal(loc=0, scale=math.sqrt(1), size=500)
        # calculemos el intervalo inferior al 95 %
        lower_bound[i] = draws.mean() - (draws.std()/math.sqrt(500)) * 1.96
        # calculemos el intervalo superior al 95 %
        upper_bound[i] = draws.mean() + (draws.std()/math.sqrt(500)) * 1.96
        # Si entre los intervalos no se contiene a 0
        if (lower_bound[i] < 0) and (upper_bound[i] > 0):
            # marcar como 1
            coverage[i] = 1
            # de lo contrario
        else:
            # marcar como 0
            coverage[i] = 0

    cnt = []
    for i, ci in enumerate(zip(lower_bound, upper_bound)):
        cnt.append(i+1)

    # guardemos la información en un dataframe para facilitar la manipulación
    coverage_range = pd.DataFrame({ 'counter': cnt,
                                'lb': lower_bound,
                                'ub': upper_bound,
                                'rejected' : coverage} )

    plt.axhline(y = 0, lw=3, color = '#1c6cab')
    plt.annotate(r'$\theta$ ', xy=(101,.01), fontsize=20, color='#1c6cab')

    for i, row in coverage_range.iterrows():
        if row['rejected'] == 1:
            plt.vlines(row['counter'], row['lb'], row['ub'], color = 'dodgerblue', linewidth = 1.5)
        else:
            plt.vlines(row['counter'], row['lb'], row['ub'], color = 'tomato', linewidth = 3)

    plt.xlabel('Iteraciones')
    plt.ylabel('Parámetro')
    plt.title('')

def significance_threshold(cutoff, c):
    xaxis = np.linspace(-3, 3, 500)
    t_distribution = stats.t.pdf(xaxis, 500)
    cutoff_point = stats.t.ppf(cutoff, 500)
    plt.plot(xaxis, t_distribution, color='#1c6cab', lw=3)
    plt.axvline(cutoff_point, 0, 0.4, color=c, 
                label=r'Sig: {0}% z: {1}'.format(int((1-(2*cutoff)) * 100), -round(cutoff_point, 2)),
                linestyle='--')
    plt.annotate("{}".format(-cutoff), xy=(cutoff_point-.25, 0.16), color=c, )
    plt.axvline(-cutoff_point, 0, 0.4, color=c, linestyle='--')
    plt.annotate("{}".format(cutoff), xy=(-cutoff_point, 0.16), color=c)
    plt.annotate("Falla en Rechazar \nHipótesis Nula", xy=(-0.35,.25))
    plt.fill_between(xaxis, 0, .4, where=xaxis > 1.62, alpha=.1, facecolor='slategrey')
    plt.fill_between(xaxis, 0, .4, where=xaxis < -1.62, alpha=.1, facecolor='slategrey')
    plt.annotate("Rechazo \nHipótesis Nula", xy=(1.96, .20))
    plt.annotate("Rechazo \nHipótesis Nula", xy=(-2.7, .20))
    plt.legend(loc = 8, fontsize=12)
    plt.ylim(0, .40)
    plt.title(r'Regiones de rechazo en la distribución de la nula $H_{0}\sim\mathcal{N}(0,1)$')
    plt.ylabel('Densidad')
    plt.xlabel('Rango')


    for i, p_value in enumerate([0.005, 0.025, 0.05]):
        significance_threshold(p_value, colors[i])

def gelman_hill_sim():
    """docstring for gelman_hill_sim"""
    birth_type = np.random.choice(['Fraternal', 'Identical', 'Single'], size=400, p=[1/125, 1/300, (1- 1/125 - 1/300)], replace=True)
    girls = np.full(400, 'NaN')
    for i in len(401):
        if birth_type[i] == "Single":
            girls[i] = np.random.binomial(1, .488, 1)
        elif birth_type[i] == "Identical":
            girls[i] = np.random.binomial(1, .495, 1)
        elif birth_type[i] == 'Fraternal':
            girls[i] = np.random.binomial(1, .495, 2)

def t_distribution(degree_freedom = [1, 5, 10, 30, 60]):
    """docstring for t_"""
    x_axis = np.linspace(-3, 3, 100)

    for i, degree in enumerate(degree_freedom):
        plt.plot(x_axis, stats.t.pdf(x_axis, degree), color=colors[i],
                linestyle = '--', lw=2, label="Grados de Libertad: {}". format(degree))

    plt.plot(x_axis, stats.norm.pdf(x_axis), color=colors[5], label = r'$X\sim\mathcal{N}(0,1)$', lw=4)
    plt.legend()


def significance_threshold(cutoff, i):
    xaxis = np.linspace(-3, 3, 500)
    t_distribution = stats.t.pdf(xaxis, 500)
    cutoff_point = stats.t.ppf(cutoff, 500)
    plt.plot(xaxis, t_distribution, color='#1c6cab', lw=3)
    plt.axvline(cutoff_point, 0, 0.4, color=colors[i],
                label=r'Sig: {0}% z: {1}'.format(int((1-(2*cutoff)) * 100), round(cutoff_point, 2)),
                linestyle='--')
    plt.annotate("{}".format(-cutoff), xy=(cutoff_point-.25, 0.16), color=colors[i], )
    plt.axvline(-cutoff_point, 0, 0.4, color=colors[i], linestyle='--')
    plt.annotate("{}".format(cutoff), xy=(-cutoff_point, 0.16), color=colors[i])
    plt.annotate("Falla en Rechazar \nHipótesis Nula", xy=(-0.35, .25))
    plt.fill_between(xaxis, 0, .4, where=xaxis > 1.62, alpha=.1, facecolor='slategrey')
    plt.fill_between(xaxis, 0, .4, where=xaxis < -1.62, alpha=.1, facecolor='slategrey')
    plt.annotate("Rechazo \nHipótesis Nula", xy=(1.96, .20))
    plt.annotate("Rechazo \nHipótesis Nula", xy=(-2.7, .20))
    plt.legend(loc=8, fontsize=12)
    plt.ylim(0, .40)
    plt.title(r'Regiones de rechazo en la distribución de la nula $H_{0}\sim\mathcal{N}(0,1)$')
    plt.ylabel('Densidad')
    plt.xlabel('Rango')

def graph_significance():
    for i, p_value in enumerate([0.005, 0.025, 0.05, 0.10]):
        significance_threshold(p_value, colors[i])



def confidence_intervals():

    """docstring for confidence_intervals"""

    sims = 100
    coverage = np.empty(sims)
    lower_bound = np.empty(sims)
    upper_bound = np.empty(sims)

    for i in range(sims):
        draws = np.random.normal(loc=0, scale=math.sqrt(1), size=500)
        lower_bound[i] = draws.mean() - (draws.std() / math.sqrt(500)) * 1.96
        upper_bound[i] = draws.mean() + (draws.std() / math.sqrt(500)) * 1.96
        if (lower_bound[i] < 0) and (upper_bound[i] > 0):
            coverage[i] = 1
        else:
            coverage[i] = 0


    coverage_range = pd.DataFrame({
        'counter': list(range(1, sims + 1, 1)),
        'lb': lower_bound,
        'ub': upper_bound,
        'rejected': coverage
    })

    plt.axhline(y = 0, lw=3, color='#1c6cab')
    plt.annotate(r'$\theta$', xy=(101, .01), fontsize=20,color='#1c6cab')

    for i, row in coverage_range.iterrows():
        if row['rejected'] == 1:
            plt.vlines(row['counter'], row['lb'], row['ub'], color='dodgerblue')
        else:
            plt.vlines(row['counter'], row['lb'], row['ub'], color='tomato')
    
    plt.xlabel('Iteraciones')
    plt.ylabel('Parámetro')
    plt.title('')

def binarize_histogram_plt(dataframe, variable):
    tmp = dataframe
    tmp['binarize'] = np.where(tmp[variable] > np.mean(tmp[variable]), 1, 0)
    hist_1 = tmp[tmp['binarize'] == 1][variable].dropna()
    hist_0 = tmp[tmp['binarize'] == 0][variable].dropna()
    plt.subplot(1, 2, 1)
    plt.hist(hist_0, color='lightgrey')
    plt.axvline(np.mean(hist_0))
    plt.title(f"{variable} <= {round(np.mean(tmp[variable]), 3)}")#.format(variable, round(np.mean(tmp[variable]), 3)))
    plt.subplot(1, 2, 2)
    plt.hist(hist_1,color='lightgrey')
    plt.axvline(np.mean(hist_1))
    plt.title(f"{variable} >= {round(np.mean(tmp[variable]), 3)}")#.format(variable, round(np.mean(tmp[variable]), 3)))

def binarize_histogram_sns(dataframe, variable):
    tmp = dataframe
    tmp['binarize'] = np.where(tmp[variable] > np.mean(tmp[variable]), 1, 0)
    hist_1 = tmp[tmp['binarize'] == 1][variable].dropna()
    hist_0 = tmp[tmp['binarize'] == 0][variable].dropna()
    plt.subplot(1, 2, 1)
    sns.histplot(data= hist_0, color='lightgrey')
    plt.axvline(np.mean(hist_0))
    plt.title(f"{variable} <= {round(np.mean(tmp[variable]), 3)}")#.format(variable, round(np.mean(tmp[variable]), 3)))
    plt.subplot(1, 2, 2)
    sns.histplot(data= hist_1,color='lightgrey')
    plt.axvline(np.mean(hist_1))
    plt.title(f"{variable} >= {round(np.mean(tmp[variable]), 3)}")#.format(variable, round(np.mean(tmp[variable]), 3)))

def hist_hipotesis(df, var, binarize):
    
    tmp=df.copy()
    tmp=tmp.dropna(subset=[var])
    
    plt.hist(tmp[tmp[binarize] ==1][var], alpha=0.4, label=binarize)
    plt.hist(tmp[tmp[binarize] ==0][var], label=f'not {binarize,}', alpha=0.4)

    plt.legend()
    plt.show()

def grouped_boxplot(dataframe, variable, group_by):
    tmp = dataframe
    stratify_by = tmp[group_by].unique()
    if len(stratify_by) / 2 > 3:
        fig, ax = plt.subplots(2, len(stratify_by),sharey=True)
    else:
        fig, ax = plt.subplots(1, len(stratify_by),sharey=True)
    for i, n in enumerate(stratify_by):
        ax[i].boxplot(tmp[tmp[group_by] == n][variable])
        ax[i].set_title(n)


def grouped_scatterplot(dataframe, x, y, group_by):
    tmp = dataframe
    stratify_by = tmp[group_by].unique()
    if len(stratify_by) / 2 > 3:
        fig, ax = plt.subplots(2, len(stratify_by),sharey=True)
    else:
        fig, ax = plt.subplots(1, len(stratify_by),sharey=True)
    for i, n in enumerate(stratify_by):
        tmp_group_plt = tmp[tmp[group_by] == n]
        ax[i].plot(tmp_group_plt[x], tmp_group_plt[y], 'o')
        ax[i].set_title(n)

'''def hist_var_bin(df, var, binarize):
    tmp = df
    tmp['binarize'] = np.where(tmp[df] > np.mean(tmp[var]), 1, 0)
    hist_1 = tmp[tmp['binarize'] == 1][var].dropna()
    hist_0 = tmp[tmp['binarize'] == 0][var].dropna()

    bins = np.linspace(0, 10, 20)

    plt.hist(x, bins, alpha = 0.5, label = binarize)
    plt.hist(y, bins, alpha = 0.5, label = f'No {binarize}')
    plt.xlabel(var)
    plt.ylabel('count')
    plt.legend(loc = 'upper left')
    plt.axvline(x.mean(), ls = '--', c = 'lightblue')
    plt.axvline(y.mean(), ls = '--', c = 'orange')
    plt.show()'''


def histogram_var_bin(dataframe, variable):

    tmp = dataframe
    tmp['binarize'] = np.where(tmp[variable] > np.mean(tmp[variable]), 1, 0)
    x = tmp[tmp['binarize'] == 1][variable].dropna()
    y = tmp[tmp['binarize'] == 0][variable].dropna()

    plt.subplot(1, 2, 1)
    plt.hist(x, alpha = 0.5, color = 'lightblue')
    plt.axvline(np.mean(x))
    plt.title(f"{variable} <= {round(np.mean(tmp[variable]), 3)}")
    plt.subplot(1, 2, 2)
    plt.hist(y,alpha = 0.5, color = 'orange')
    plt.axvline(np.mean(y))
    plt.title(f"{variable} >= {round(np.mean(tmp[variable]), 3)}")

def summary_drop(data):
    tipos = pd.DataFrame({'tipo': data.dtypes},index=data.columns)
    na = pd.DataFrame({'nulos': data.isna().sum()}, index=data.columns)
    na_prop = pd.DataFrame({'porc_nulos':data.isna().sum()/data.shape[0]}, index=data.columns)
    ceros = pd.DataFrame({'ceros':[data.loc[data[col]==0,col].shape[0] for col in data.columns]}, index= data.columns)
    ceros_prop = pd.DataFrame({'porc_ceros':[data.loc[data[col]==0,col].shape[0]/data.shape[0] for col in data.columns]}, index= data.columns)
    summary = data.describe(include='all').T

    summary['dist_IQR'] = summary['75%'] - summary['25%']
    summary['limit_inf'] = summary['25%'] - summary['dist_IQR']*1.5
    summary['limit_sup'] = summary['75%'] + summary['dist_IQR']*1.5

    summary['outliers'] = data.apply(lambda x: sum(np.where((x<summary['limit_inf'][x.name]) | (x>summary['limit_sup'][x.name]),1 ,0)) if x.name in summary['limit_inf'].dropna().index else 0)


    return pd.concat([tipos, na, na_prop, ceros, ceros_prop, summary], axis=1).sort_values('tipo')

def cross_plot(data, barra, variable, categorias, size=(10,7), xlim=(-0.5,3.5), ylim=(0.1,0.8), titulo = None, order=1, medias=0):
    fig, ax1 = plt.subplots(figsize=size)
    ax2 = ax1.twinx()
    if order==1:
        data = data.sort_values(barra).reset_index(drop=True)
    data[barra].plot(kind='bar', color='b', ax=ax1, label=barra)
    try:
        for v in variable:
            data[v].plot(kind='line', marker='d', ax=ax2, label=v)
    except:
        data[variable].plot(kind='line',color='r', marker='d', ax=ax2, label=variable)
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()
    ticks = data[categorias]
    plt.xticks(np.arange(ticks.unique().shape[0]),ticks)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if medias==1:
        cc = ['blue','red','gray']
        j=0
        try:
            for v in variable:
                plt.axhline(data[v].mean(), label=v, color=cc[j])
                j=j+1
        except:
            plt.axhline(data[variable].mean(), label=variable, color=cc[j])
    plt.title(titulo)
    ax1.set_xlabel(categorias)
    ax1.set_ylabel(barra)
    ax2.set_ylabel(variable)
    plt.legend()
    plt.grid()
    fig.tight_layout()  
    plt.show()

def form_model(df, var_obj):
    """Modelo logit con todos sus atributos.
    Args:
        df (dataframe): Conjunto de datos
        var_obj (string): variable objetivo
    Returns:
        string: formula del modelo
    """
    base_formula = f'{var_obj} ~ '
    for col in df.columns:
        if col != var_obj:
            base_formula += f'{col} + '
    return base_formula[:-3]

def predict(df, var_obj):
        """Función que automatiza las predicciones por LogisticRegression.

        Args:
                df (dataframe): dataframe con todas las variables a introducir en el modelo, incluida la V.O
                var_obj (str): variable objetivo

        Returns:
                array: vector de prueba y vector de predicciones
        """
        # separando matriz de atributos de vector objetivo
        # utilizamos dataframe con variables significativas
        mat_atr = df.drop(var_obj, axis=1)
        vec_obj = df[var_obj]
        # split de conjuntos de entrenamiento vs prueba
        X_train, X_test, y_train, y_test = train_test_split(mat_atr, vec_obj, test_size = .33, random_state = 15820)
        # estandarizamos conjunto de entrenamiento
        X_train_std = StandardScaler().fit_transform(X_train)
        X_test_std = StandardScaler().fit_transform(X_test)
        # ajustamos modelo sin alterar hiperparámetros
        modelo_x =  LogisticRegression().fit(X_train_std, y_train)
        # prediccion de clases y probabilidad
        y_hat = modelo_x.predict(X_test_std)
        return modelo_x, y_test, y_hat

def report_scores(y_predict, y_validate):
    """Calcula el error cuadrático medio y el r2 score entre dos vectores. El primero, el vector de valores predecidos por el
    conjunto de prueba, y el segundo, el vector objetivo original.

    Args:
        y_predict (vector): vector de valores predecidos
        y_validate (vector): vector de valores verdaderos
    """
    mse = mean_squared_error(y_validate, y_predict)
    r2 = r2_score(y_validate, y_predict).round(2)
    print(f'Error cuadrático medio: {mse}')
    print(f'R2: {r2}')

def get_vars4type(df, exception, var='all'):
    """
    Funcion que, dado un DataFrame de entrada, devuelve las variables segmentadas en numericas y/o categorias.
    
    Entrada:
        - df        : Dataframe que se desea analizar.
        - exception : Listado de excepciones en formato lista.  
                    Las excepciones son los nombres de las columnas del DataFrame "df"
        - var       : el tipo de salida de la funcion en formato lista.
        - var='cat': Entrega las variables consideradas como categoricas.
        - var='num': Entrega las variables consideradas como numericas.
        - var='all': Entrega todas las variables
        
    Salida:
        La funcion entrega su salida en formato lista de la siguiente forma:
        
        ['var1', 'var2', 'var3', ... , 'varn']
    """
    var_num = []
    var_cat = []
    for i in df.columns:
        if i not in exception:
            if pd.api.types.is_numeric_dtype(df[i]):
                var_num.append(i)
            else:
                var_cat.append(i)
    
    if var == 'cat':
        return var_cat
    elif var == 'num':
        return var_num
    else:
        return var_cat + var_num


def get_describe(df, num=True):
    """
    Funcion que entrega las medidas descriptivas de las variables que componen un DataFrame.
    La funcion identifica el tipo de variable y entrega la descripcion de esta.
    Ademas, la funcion entrega los graficos de estas medidas en formato distplot, para las 
    medidas numericas y countplot para las medidas no numericas.
    
    Entrada:
        - df        : Dataframe que se desea analizar.
        - num       : Variable binaria que define la salida de la funcion.
         - num=True : Entrega la descripcion y graficos de las variables numericas.
         - num=False: Entrega los .value_counts() y graficos de las variables no numericas.
         
    Salida:
                variable_name          variable_name_2
                count       <value>            <value>
                mean        <value>            <value>
                std         <value>            <value>
                min         <value>            <value>
                25%         <value>            <value>
                50%         <value>            <value>
                75%         <value>            <value>
                max         <value>            <value>
         
                [-----------]    [-----------]
                [--grafico--]    [--grafico--]
                [-----------]    [-----------]
                
                [-----------]    [-----------]
                [--grafico--]    [--grafico--]
                [-----------]    [-----------]
    """
    var_num = []
    var_cat = []
    for i in df.columns:
            if pd.api.types.is_numeric_dtype(df[i]):
                var_num.append(i)
            else:
                var_cat.append(i)
    
    if num:
        

        print("Informacion sobre variables numericas:\n")
        print(df.loc[:,var_num].describe())
        
        plt.figure(figsize=(12,10))
        for n, i in enumerate(var_num):
            n += 1
            plt.subplot(round(len(var_num)/2,0), 2, n)
            sns.distplot(df[i])
            plt.axvline(df[i].mean(), color='tomato', linestyle='-', lw=2)
            plt.title(i)
            plt.xlabel("")
    else:        
        
        print("\n\nInformacion sobre variables categoricas:\n")

        for i in var_cat:
            print(i, "\n--------------------")
            print(df[i].value_counts('%'), "\n")
        
        plt.figure(figsize=(15,40))
        long = len(var_cat)
        for n, i in enumerate(var_cat):
            n += 1
            plt.subplot(long, 1, n)
            sns.countplot(y=df[i], order=df[i].value_counts().index)
            plt.title(i)
            plt.xlabel("")

    plt.tight_layout()

    