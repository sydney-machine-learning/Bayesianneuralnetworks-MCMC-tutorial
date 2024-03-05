import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from scipy.stats import mode

def histogram_trace(pos_points, true_posterior=None, burn_in=None, fname = None, **kwargs):
    '''
    This function will create a histogram and traceplot of the MCMC results.
    ''' 
    size = 15

    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(10, 4))
    if not burn_in is None:
        plot_points = pos_points[burn_in:,...]
    else:
        plot_points = pos_points

    ax1 = fig.add_subplot(111)
    ax1.hist(plot_points,  bins = 20, color='C0', alpha=0.7, label='Sampled posterior')
    x_lims = ax1.get_xlim()
    if not true_posterior is None:
        ax2 = ax1.twinx()
        ax2.grid(False)
        ax2.plot(true_posterior[:,0], true_posterior[:,1], linewidth=2, color='C1', label='True\nDistribution')
        ax2.set_ylim(0, ax2.get_ylim()[1])
        ax2.set_ylabel('Density', fontsize = size)
    ax1.set_xlim(x_lims)
    ax1.set_ylabel('Frequency', fontsize = size, labelpad=10)
    ax1.set_xlabel(kwargs.get('param_name','Parameter value'), fontsize = size, labelpad=10)
    ax1.set_title(kwargs.get('title','Posterior'), fontsize = size, pad=10)
    lgd=plt.legend(bbox_to_anchor=(1.25,0.5),loc='center left')
    fig.tight_layout()
    if not fname is None: 
        plt.savefig(fname + '_posterior.pgf', 
            bbox_extra_artists=(lgd,), 
            bbox_inches='tight',
            dpi=300
        )
        plt.clf()
    else:
        plt.show()

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(111)
    if not burn_in is None:
        ax1.plot(np.arange(burn_in), pos_points[:burn_in], color='C3',label='Burn-in trace')
        ax1.plot(np.arange(burn_in,pos_points.shape[0]), pos_points[burn_in:], color='C0',label='Posterior trace')
    else:
        ax1.plot(pos_points,label='Posterior trace')   
    lgd = plt.legend(loc='center left',bbox_to_anchor=(1.025,0.5))
    plt.title("Parameter trace plot", fontsize = size, pad=10)
    plt.xlabel(' Number of Samples  ', fontsize = size, labelpad=10)
    plt.ylabel(' Parameter value ', fontsize = size, labelpad=10)
    plt.tight_layout()
    if not fname is None:
        plt.savefig(
            fname + '_trace.pgf', 
            bbox_extra_artists=(lgd,), 
            bbox_inches='tight',
            dpi=300
        )
        plt.clf()
    else:
        plt.show()

def plot_ycorr_scatter(y_obs,y_mod,minmax=(0,1),fname=None,dy=False):
    sns.set_context("talk")
    sns.set_style("ticks",{'axes.grid': True})

    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_subplot(111)

    if not minmax is None:
        ax1.set_xlim(minmax[0],minmax[1])
        ax1.set_ylim(minmax[0],minmax[1])
    else:
        ax1.set_xlim(np.min([y_obs.min(),y_mod.min()]),np.max([y_obs.max(),y_mod.max()]))
        ax1.set_ylim(np.min([y_obs.min(),y_mod.min()]),np.max([y_obs.max(),y_mod.max()]))
    # plot red dashed 1:1 line
    ax1.plot(ax1.get_xlim(),ax1.get_ylim(),'--r')

    sns.scatterplot(
        x=np.mean(y_mod,axis=0).squeeze(),y=y_obs.squeeze(),ax=ax1,
    )

    if not dy:
        ax1.set_xlabel('Y modelled')
        ax1.set_ylabel('Y observed')
    else:
        ax1.set_xlabel('$\Delta$Y modelled')
        ax1.set_ylabel('$\Delta$Y observed')

    if not fname is None:
        plt.savefig(
            fname,
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
    else:
        plt.show()

def plot_y_timeseries(y_obs,y_mod,y_sim=None,dataset_name=None,ci=False,fname=None):
    sns.set_context("talk")
    sns.set_style("ticks",{'axes.grid': True})

    fig = plt.figure(figsize=(9,4))
    ax1 = fig.add_subplot(111)

    x = np.arange(y_obs.shape[0])

    sns.lineplot(
        x=x,y=y_obs.squeeze(),
        ax=ax1,
        label='Observed'
    )
    sns.lineplot(
        x=x,y=np.mean(y_mod,axis=0),
        ax=ax1,color='C3',
        label='Modelled'
    )
    if ci:
        ax1.fill_between(
            x,
            np.percentile(y_mod,2.5,axis=0),
            np.percentile(y_mod,97.5,axis=0),
            color='C3',alpha=0.2,
            label='Modelled 95% CI')
        if not y_sim is None:
            ax1.fill_between(
                x,
                np.percentile(y_sim,2.5,axis=0),
                np.percentile(y_sim,97.5,axis=0),
                color='C1',alpha=0.2,
                label='Simulated 95% CI')
    else:
        ax1.fill_between(
            x,
            np.mean(y_mod,axis=0)-2*np.std(y_mod,axis=0),
            np.mean(y_mod,axis=0)+2*np.std(y_mod,axis=0),
            color='C3',alpha=0.2,
            label='Modelled $\pm 2\sigma$')
        if not y_sim is None:
            ax1.fill_between(
                x,
                np.mean(y_sim,axis=0)-2*np.std(y_sim,axis=0),
                np.mean(y_sim,axis=0)+2*np.std(y_sim,axis=0),
                color='C1',alpha=0.2,
                label='Simulated $\pm 2\sigma$')
    
    if dataset_name is not None:
        ax1.set_title('{} Data'.format(dataset_name))
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Y')
    lgd = plt.legend(loc='center left',bbox_to_anchor=(1.05,0.5))
    if not fname is None:
        plt.savefig(
            fname, 
            bbox_extra_artists=(lgd,), 
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
    else:
        plt.show()

def plot_linear_data(x,y,y_modelled=None,y_simulated=None,ci=False,save_fig=False):
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
            np.percentile(y_simulated,2.5,axis=0),
            np.percentile(y_simulated,97.5,axis=0),
            color='orange',alpha=0.2,
            label='Simulated 95% CI')
    else:
        ax1.fill_between(
            x.squeeze(),
            np.mean(y_simulated,axis=0)-2*np.std(y_simulated,axis=0),
            np.mean(y_simulated,axis=0)+2*np.std(y_simulated,axis=0),
            color='orange',alpha=0.2,
            label='Simulated $\pm 2\sigma$')

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
    lgd = plt.legend(loc='center',bbox_to_anchor=(1.3,0.5))
    if save_fig:
        fig.savefig(os.path.join('.', 'figures', 'linear_model_fit.png'),bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(obs_y, pred_y, title='Confusion matrix', cmap=plt.cm.Blues):
    sns.set_context("talk")
    sns.set_style("ticks",{'axes.grid': True})

    fig = plt.figure(figsize=(5,4))
    ax1 = fig.add_subplot(111) 

    sns.heatmap(
        confusion_matrix(obs_y, mode(pred_y, axis=0)[0]),
        annot=True, fmt='.0f', cmap=cmap, ax=ax1,
        cbar=False
    )
    # print(confusion_matrix(obs_y, mode(pred_y, axis=0)[0]))
    ax1.set_title(title)
    ax1.set_xlabel('Predicted label')
    ax1.set_ylabel('Observed label')
    plt.show()
    
def boxplot_weights(results, width=20,skip=2):
    '''
    Visualise the weights of the Bayesian Neural Network
    '''
    fig = plt.figure(figsize=(width,width*0.3))
    ax1 = fig.add_subplot(111)

    df = pd.melt(results.drop(columns=['rmse'],errors='ignore'))
    sns.boxplot(data=df,x='variable',y='value', ax=ax1)
    # set labels as invisible to help clutter
    for label in ax1.xaxis.get_ticklabels()[1::skip]:
        label.set_visible(False)
    ax1.set_ylabel('Posterior')
    #         plt.legend(loc='upper right')
    ax1.set_title("Boxplot of Posterior W (weights and biases)", pad=20)
    ax1.set_xlabel('$[w_h][w_o][b_h][b_o][tau]$')
    plt.show()