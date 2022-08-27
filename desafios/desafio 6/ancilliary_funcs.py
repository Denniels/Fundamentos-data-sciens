# Importacion de librerias necesarias
#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

def fetch_descriptives(dataframe):
    for key, value in dataframe.iteritems():
        print(value.describe())


def fetch_null_cases(dataframe, var, print_list=False):
    tmp = dataframe
    tmp['flagnull'] = tmp[var].isnull()
    count_na = 0
    for i, r in tmp.iterrows(): 
        if r['flagnull'] is True:
            count_na += 1
            if print_list is True:
                print( r['cname'])
    print(f'''
    Casos perdidos para {var}
    Cantidad de Casos: {count_na}
    Porcentaje de la muestra {(count_na/len(tmp)) * 100}''')

    if print_list is True:
        print("Países sin registros de {0}\n".format(var))

def plot_hist(sample_df, full_df, var, sample_mean=False, true_mean=False):
    tmp = sample_df[var].dropna()
    plt.hist(tmp, color='grey', alpha=.4)
    plt.title(var)
    if sample_mean is True: 
        plt.axvline(np.mean(tmp), color='dodgerblue')
    if true_mean is True: 
        plt.axvline(np.mean(full_df[var]), color='tomato')

def plot_histograma(df,columna, df_total,media_df_peque= False, media_df_total= False):
    temporal = df[columna].dropna()
    plt.hist(temporal, color='grey')
    plt.grid(True)
    plt.title(columna)
    if media_df_peque is True:
        plt.axvline(np.mean(temporal), color='dodgerblue')
    if media_df_total is True:
        plt.axvline(np.mean(df_total[columna]), color='red')
    

def dotplot(df, plot_var, plot_by, global_stat = False, statistic = 'mean'):
    tmp_df = df.loc[:, [plot_by, plot_var]]
    if statistic is 'mean':
        tmp_group_stat = tmp_df.groupby(plot_by)[plot_var].mean()
    if statistic is 'median':
        tmp_group_stat = tmp_df.groupby(plot_by)[plot_var].median()
    plt.plot(tmp_group_stat.values, tmp_group_stat.index, 'o', color='grey')
    if global_stat is True and statistic is 'mean': 
        plt.axvline(df[plot_var].mean(), color='tomato', linestyle='--')
    if global_stat is True and statistic is 'median': 
        plt.axvline(df[plot_var].median(), color='tomato', linestyle='--')
    
def dotplot_1(df,plot_var,plot_by,global_stat=False,statistics="mean"):
    if global_stat:
        plt.axvline(np.mean(df)[plot_var])
        
    if statistics=="mean":
        grouped= df.groupby(plot_by)[plot_var].agg("mean")
    elif statistics=="median":   
        grouped= df.groupby(plot_by)[plot_var].agg("median")
    else:
        df["z"]= (df[plot_var]-df[plot_var].mean())/ df[plot_var].std()
        grouped= df.groupby(plot_by)["z"].agg("mean")
        plt.axvline(0)
        
    plt.plot(grouped.values, grouped.index, "o", color="grey")
    plt.title(plot_var)    
    
    plt.show()