import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Any


def correlation_matrix(df,precision=2,plot=False):

    """
    Returns correlation matrix of a pandas DataFrame
    optionally plots it.
    """
    
    df_corr = df.corr()
    
    if plot:
        fig = plt.figure(figsize=(9,7))
        fig.suptitle('Correlation Matrix')
        sns.heatmap(df_corr, annot=True)
        plt.show()
    return df_corr

def get_principle_component(df,least_components=None,least_explained_variance=None,plot=True):
    """
    Return a dictionary of the principle components of the given DataFrame
    n_components: Number of principle components
    explained_variance: Percentage of variance to look for 
                        (say 0.85, means to find the no of principle components that 
                        can explain 85% of the variance in the data)
    {'col1':1, 'col2':2, 'col3':3}
    """
    if not (least_components or least_explained_variance):
        raise ValueError('n_components or explained_variance can not be empty')

    
    # Loop Function to identify number of principal components that explain at least 85% of the variance
    for comp in range(least_components, df.shape[1]):
        pca = PCA(n_components= comp, random_state=42)
        pca.fit(scale(df))
        comp_check = pca.explained_variance_ratio_
        final_comp = comp
        if comp_check.sum() > least_explained_variance:
            break
    print("Using {} components, we can explain {}% of the variability in the original data.".format(final_comp,round(comp_check.sum()*100,2)))

    Final_PCA = PCA(n_components=final_comp,random_state=42)
    Final_PCA.fit(scale(df))
 
    if plot:
        bar_range = final_comp+1

        fig, ax1 = plt.subplots(figsize=(9,7))
        # ax1 = fig.add_subplot(211)
        ax2 = ax1.twinx()

        fig.suptitle('Principle component Analysis')


        ax1.bar(list(range(1, bar_range)),Final_PCA.explained_variance_ratio_,color='g',label="% of variance explained")
        ax1.legend(loc='upper right')


        ax2.plot(list(range(1, bar_range)), np.cumsum(Final_PCA.explained_variance_ratio_), color='r',label="% of cumulative variance explained")
        ax2.legend(loc='upper left')

        # for ax in [ax1,ax2]:
        ax1.set_xlim([1,bar_range])
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0,decimals=1))
        ax1.set_xlabel('Principle components')
        ax1.set_ylabel('Explained variance')
        ax2.set_yticklabels([])
        plt.setp(ax1, xticks=[s for s in range(1, bar_range)])
            

        plt.show()

   
    return Final_PCA,final_comp