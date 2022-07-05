import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import seaborn as sns
import numpy as np

def histogram_trace(pos_points, fname = None):
    '''
    This function will create a histogram and traceplot of the MCMC results.
    ''' 
    size = 15

    plt.tick_params(labelsize=size)
    params = {'legend.fontsize': size, 'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.grid(alpha=0.75)

    plt.hist(pos_points,  bins = 20, color='#0504aa', alpha=0.7)   
    plt.title("Posterior distribution ", fontsize = size)
    plt.xlabel(' Parameter value  ', fontsize = size)
    plt.ylabel(' Frequency ', fontsize = size) 
    plt.tight_layout()
    if not fname is None: 
        plt.savefig(fname + '_posterior.png')
        plt.clf()
    else:
        plt.show()


    plt.tick_params(labelsize=size)
    params = {'legend.fontsize': size, 'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.grid(alpha=0.75) 
    plt.plot(pos_points)   

    plt.title("Parameter trace plot", fontsize = size)
    plt.xlabel(' Number of Samples  ', fontsize = size)
    plt.ylabel(' Parameter value ', fontsize = size)
    plt.tight_layout()
    if not fname is None:
        plt.savefig(fname + '_trace.png') 
        plt.clf()
    else:
        plt.show()

def plot_ycorr_scatter(y_obs,y_mod):
    sns.set_context("talk")
    sns.set_style("ticks",{'axes.grid': True})

    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_subplot(111)
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
