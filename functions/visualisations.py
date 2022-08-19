import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import seaborn as sns
import numpy as np
import pandas as pd

def histogram_trace(pos_points, true_posterior=None, burn_in=None, fname = None):
    '''
    This function will create a histogram and traceplot of the MCMC results.
    ''' 
    size = 15

    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(7, 5))
    if not burn_in is None:
        plot_points = pos_points[burn_in:,...]
    else:
        plot_points = pos_points

    ax1 = fig.add_subplot(111)
    ax1.hist(plot_points,  bins = 20, color='C0', alpha=0.7)
    x_lims = ax1.get_xlim()
    if not true_posterior is None:
        ax2 = ax1.twinx()
        ax2.grid(False)
        ax2.plot(true_posterior[:,0], true_posterior[:,1], linewidth=2, color='C1', label='True Distribution')
        ax2.set_ylim(0, ax2.get_ylim()[1])
    ax1.set_xlim(x_lims)
    ax1.set_ylabel('Frequency')
    plt.title("Frequency ", fontsize = size)
    plt.xlabel(' Parameter value  ', fontsize = size)
    plt.ylabel(' Density ', fontsize = size) 
    plt.tight_layout()
    if not fname is None: 
        plt.savefig(fname + '_posterior.png')
        plt.clf()
    else:
        plt.show()

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(111)
    if not burn_in is None:
        ax1.plot(np.arange(burn_in), pos_points[:burn_in], color='C2',label='Burn in')
        ax1.plot(np.arange(burn_in,pos_points.shape[0]), pos_points[burn_in:], color='C0',label='Posterior')
    else:
        ax1.plot(pos_points)   
    plt.legend(loc='center',bbox_to_anchor=(1.15,0.5))
    plt.title("Parameter trace plot", fontsize = size)
    plt.xlabel(' Number of Samples  ', fontsize = size)
    plt.ylabel(' Parameter value ', fontsize = size)
    plt.tight_layout()
    if not fname is None:
        plt.savefig(fname + '_trace.png') 
        plt.clf()
    else:
        plt.show()

def plot_ycorr_scatter(y_obs,y_mod,minmax=True):
    sns.set_context("talk")
    sns.set_style("ticks",{'axes.grid': True})

    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_subplot(111)

    if minmax:
        ax1.set_xlim(0,1)
        ax1.set_ylim(0,1)
        # plot red dashed 1:1 line
        ax1.plot(ax1.get_xlim(),ax1.get_ylim(),'--r')

    sns.scatterplot(
        x=np.mean(y_mod,axis=0).squeeze(),y=y_obs.squeeze(),ax=ax1,
    )

    ax1.set_xlabel('y modelled')
    ax1.set_ylabel('y observed')
    plt.show()

def plot_y_timeseries(y_obs,y_mod,dataset_name=None,ci=False):
    sns.set_context("talk")
    sns.set_style("ticks",{'axes.grid': True})

    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_subplot(111)

    x = np.arange(y_obs.shape[0])

    sns.lineplot(
        x=x,y=y_obs.squeeze(),
        ax=ax1,
        label='Observed'
    )
    sns.lineplot(
        x=x,y=np.mean(y_mod,axis=0),
        ax=ax1,color='red',
        label='Modelled'
    )
    if ci:
        ax1.fill_between(
            x,
            np.percentile(y_mod,2.5,axis=0),
            np.percentile(y_mod,97.5,axis=0),
            color='red',alpha=0.2,
            label='Modelled 95% CI')
    else:
        ax1.fill_between(
            x,
            np.mean(y_mod,axis=0)-2*np.std(y_mod,axis=0),
            np.mean(y_mod,axis=0)+2*np.std(y_mod,axis=0),
            color='red',alpha=0.2,
            label='Modelled $\pm 2\sigma$')
    if dataset_name is not None:
        ax1.set_title('{} Data'.format(dataset_name))
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Y')
    plt.legend(loc='center',bbox_to_anchor=(1.3,0.5))
    plt.show()

def plot_linear_data(x,y,y_modelled=None,ci=False):
    '''
    This function will plot the data in linear format.
    '''
    sns.set_context("talk")
    sns.set_style("ticks",{'axes.grid': True})

    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_subplot(111)

    sns.scatterplot(
        x=x.squeeze(),y=y.squeeze(),ax=ax1,
        label='Data'
    )
    
    sns.lineplot(
        x=x.squeeze(),y=np.mean(y_modelled,axis=0),
        ax=ax1,color='red',
        label='Mean Modelled'
    )

    if ci:
        ax1.fill_between(
            x.squeeze(),
            np.percentile(y_modelled,2.5,axis=0),
            np.percentile(y_modelled,97.5,axis=0),
            color='red',alpha=0.2,
            label='Modelled 95% CI')
    else:
        ax1.fill_between(
            x.squeeze(),
            np.mean(y_modelled,axis=0)-2*np.std(y_modelled,axis=0),
            np.mean(y_modelled,axis=0)+2*np.std(y_modelled,axis=0),
            color='red',alpha=0.2,
            label='Modelled $\pm 2\sigma$')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.legend(loc='center',bbox_to_anchor=(1.3,0.5))
    plt.show()


def boxplot_weights(results, width=20,skip=2):
    '''
    Visualise the weights of the Bayesian Neural Network
    '''
    fig = plt.figure(figsize=(width,width*0.3))
    ax1 = fig.add_subplot(111)

    df = pd.melt(results.drop(columns=['rmse']))
    sns.boxplot(data=df,x='variable',y='value', ax=ax1)
    # set labels as invisible to help clutter
    for label in ax1.xaxis.get_ticklabels()[1::skip]:
        label.set_visible(False)
    ax1.set_ylabel('Posterior')
    #         plt.legend(loc='upper right')
    ax1.set_title("Boxplot of Posterior W (weights and biases)", pad=20)
    ax1.set_xlabel('$[w_h][w_o][b_h][b_o][tau]$')
    plt.show()