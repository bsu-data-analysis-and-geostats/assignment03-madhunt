#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def moving_avg_weighted(x_data, y_data, x_eval, win):
    '''
    Uses a weighted moving average (non-parametric smooth) to smooth a 
    dataset of one variable.
    INPUTS
        x_data  : np array, size nx1    : Independent variable
        y_data  : np array, size nx1    : Dependent variable
        x_eval  : np array, size mx1    : Locations to evaluate model
        win     : int                   : Window size (half of window)
    RETURNS
        y_model : np array, size mx1    : Non-parametric smooth estimates
    '''
    # initialize model values
    y_model = np.full(shape=np.shape(x_eval), fill_value=np.nan)

    for i in range(len(x_eval)):
        # calculate location between point and all other data points
        dist = np.sqrt( (x_data - x_eval[i])**2 )
        # find points within window size
        idx = np.where(dist < win)
        # there are at least some data points
        if len(idx) > 0:
            # define weights as bi-squared kernel 
            weights = 15/16 * (1 - (dist[idx] / win)**2 )**2
            # calculate estimate for point i
            y_model[i] = np.sum(weights * y_data[idx]) / np.sum(weights)
    return y_model


def polyfit_rmse(x_train, y_train, x_test, y_test, deg):
    '''
    Calculate RMSE and model for a polynomial.
    INPUTS
        x_train : np array  : Data to create polynomial model.
        y_train : np array  : Data to create polynomial model.
        x_test  : np array  : Data to evaluate polynomial model.
        y_test  : np array  : Data to compare with evaluated model.
        deg     : int       : Degree of polynomial.
    RETURNS
        y_model : np array  : Polynomial of degree deg fit to x_test.
        rmse    : float     : RMSE of model with y_test data.
    '''
    # find polynomial
    p = np.polyfit(x_train, y_train, deg=deg)
    # fit polynomial to data
    y_model = np.polyval(p, x_test)
    # calculate root-mean-square error
    rmse = root_mean_square_error(y_model, y_test)
    return y_model, rmse


def root_mean_square_error(model, measurements):
    '''
    Calculates Roor Mean Square Error.
    INPUTS
        model           : np array  : Y-values of model.
        measurements    : np array  : Y-values of actual data.
    RETURNS
        rmse            : float     : RMSE of model to measurements.
    '''
    assert len(model) == len(measurements)
    N = len(measurements)
    rmse = np.sqrt(1/N * np.sum((model - measurements)**2))
    return rmse


def monte_carlo_polyfit(data, n_samp, n_iters, deg_list):
    '''
    Perform Monte Carlo simulation by taking random samples of dataset.
    Fits polynomials of degrees in deg_list.
    INPUTS
        data        : np array, 1xn        : Dataset to perform Montel Carlo simulation on.
        n_samp      : int                  : Number of samples; sample size of simulation data.
        n_iters     : int                  : Number of iterations; number of times to run simulation.
        deg_list    : list of int          : List of degrees to perform simulation for.
    RETURNS
        mc_stats    : dict of list, 4x1xn : Dictionary containing mean, standard deviation, minimum, 
            and maximum values for the sample dataset at each iteration for each degree.
    '''
    mc_stats = pd.DataFrame(index=deg_list, columns=['RMSE Mean', 'RMSE Stdev', 'Param Mean', 'Param Stdev'])
    for deg in deg_list:

        deg_stats = {'Model Parameters': [], 'RMSE': []}

        for i in range(n_iters):
            # make a random selection from the entire dataset
            idx_samp = np.random.choice(data.shape[0], size=n_samp, replace=True)
            sample = data[idx_samp]
            # fit polynomial to data and calculated rms
            p = np.polyfit(sample[:,0], sample[:,1], deg=deg)
            # fit polynomial to data
            y_model = np.polyval(p, sample[:,0])
            # calculate root-mean-square error
            rmse = root_mean_square_error(y_model, sample[:,1])
            deg_stats['Model Parameters'].append(p)
            deg_stats['RMSE'].append(rmse)
        
        # save mean/stdev of model parameters for each deg
        mc_stats.loc[deg, 'RMSE Mean'] = np.mean(deg_stats['RMSE'])
        mc_stats.loc[deg, 'RMSE Stdev'] = np.std(deg_stats['RMSE'])
        mc_stats.loc[deg, 'Param Mean'] = [np.mean(Ai) for Ai in zip(*deg_stats['Model Parameters'])]
        mc_stats.loc[deg, 'Param Stdev'] = [np.std(Ai) for Ai in zip(*deg_stats['Model Parameters'])]

    return mc_stats


def calc_stats(data):
    '''
    Calculate statistics for a dataset, excluding nans. 
        Includes min, max, mean, and standard deviation.
    INPUTS
        data    : np array, 1xn : Array of data values. 
    RETURNS 
        stats   : dict          : Dictionary with general statistics. 
    '''
    stats = {}
    stats['Mean'] = np.nanmean(data)
    stats['Standard Deviation'] = np.nanstd(data)
    stats['Minimum'] = np.nanmin(data)
    stats['Maximum'] = np.nanmax(data)
    return stats


def calc_hist(data, bins):
    '''
    Calculate histogram (in counts) and bin centers for specified dataset.
    Uses default 30 bins for this HW assignment.
    INPUTS
        data        : np array, 1xn     : Array of data values.
        bins        : int               : Number of bins desired for histogram.
    RETURNS
        hist        : np array, 1x30    : Array of histogram counts.
        bin_centers : np array, 1x30    : Array of bin center locations along x-axis.
        width       : float             : Width of bins along x-axis.
    '''
    # calculate histogram
    hist, bin_edges = np.histogram(data, bins=bins)

    # shift bin edges to bin centers
    width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + width/2
    return hist, bin_centers, width


def plot_relative_density_hist(data, bins, ax, color, title=None):
    '''
    Plot relative density histogram on specified axis for given dataset. 
    Normalizes histogram by dividing counts by total histogram area. 
    INPUTS
        data    : np array, 1xn : Array of data values.
        color   : str           : Color to plot histogram bars.
        ax      : pyplot handle : Handle to axis to plot histogram on (e.g. subplot axis).
        title   : str           : Optional title for plot. Defaults to "Relative Density Histogram".
    RETURNS
        Relative density histogram will be plotted on given axis.
    '''
    # calculate histogram
    hist, bin_centers, width = calc_hist(data, bins)

    # normalize histogram
    hist_norm = hist / np.sum(hist * width)

    # plot on provided axis
    ax.bar(bin_centers, hist_norm, width=width, color=color)
    # label subfigure and y-axis
    ax.set_ylabel('Density')
    if title != None:
        ax.set_title(title)
    else:
        ax.set_title('Relative Density Histogram')
    return


def gaussian(x, mean, stdev):
    '''
    Calculate Gaussian distribution from given mean and standard deviation.
    INPUTS
        x       : np array, 1xn : Evenly-spaced array of x-values to calculate Gaussian over.
        mean    : float         : Mean of sample dataset.
        stdev   : float         : Standard deviation of sample dataset
    RETURNS
        f       : np array, 1xn : Array of Gaussian evaluated at x-values.
    '''
    f = 1/(stdev*np.sqrt(2*np.pi)) * np.exp(-(x-mean)**2 / (2*stdev**2))
    return f


def plot_gaussian(stats, ax):
    '''
    Plot a Gaussian on the given axis. 
    INPUTS
        stats   : dict  : Dictionary of stats for dataset, containing Mean,
            Standard Deviation, Minimum, and Maximum.
        ax      : pyplot handle : Handle to axis to plot histogram on (e.g. subplot axis).
    RETURNS
        Gaussian curve will be plotted in red on given axis.
    '''
    xvals = np.arange(stats['Minimum'],stats['Maximum'], 0.1)
    # (6) plot gaussian
    ax.plot(xvals, gaussian(xvals,stats['Mean'],stats['Standard Deviation']),
            'r-', label="Gaussian")
    return


def plot_hist_subplots(dataframe, bins, title, x_label, pathfig):
    '''
    Plot histogram subplots.
    INPUTS
        dataframe    : pd dataframeframe
    '''
    n = len(dataframe.columns)
    fig, ax = plt.subplots(1, n, 
                        tight_layout=True, sharex=True, sharey=True,    
                        figsize=[10,4])
    # create horizontal boxplot for each dataset
    colors = plt.cm.rainbow(np.linspace(0, 1, n))
    for i, col in enumerate(dataframe.columns):
        # calculate and normalize histogram
        hist, bin_centers, width = calc_hist(dataframe.iloc[:,i], bins)
        hist_norm = hist / np.sum(hist * width)
        # plot histogram
        ax[i].bar(bin_centers, hist_norm, width=width, color=colors[i],
                  label=f"Mean: {np.round(np.mean(dataframe.iloc[:,i]), 2)}")
        ax[i].set_xlabel(x_label)
        ax[i].set_title(col)
        ax[i].legend(loc='upper right')
    ax[0].set_ylabel('Density')
    fig.suptitle(title, fontsize=14)
    # save figure
    plt.savefig(pathfig, dpi=500, bbox_inches='tight')
    plt.close()
    return


def plot_moving_avg(x_data, y_data, win_list, x_label, y_label, title, pathfig):
    '''
    Calculate and plot unweighted moving average.
    '''
    # create plot
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    # plot raw data points
    ax.plot(x_data, y_data, 'k.')

    colors = plt.cm.rainbow(np.linspace(0,1,len(win_list)))
    for i, win in enumerate(win_list):
        # calculate moving avg
        y_model = pd.DataFrame(index=x_data, data=y_data)
        y_model = y_model.rolling(window=win, 
                                  center=True,          # center window in data
                                  min_periods=1).mean()        # shorten window at edges
                                  #win_type=).mean() 
        ax.plot(x_data, y_model,
                '-', color=colors[i], 
                label=f"{win} m window")
    # figure formatting
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc='lower left', title='Legend')
    ax.grid(True)
    fig.suptitle(title)
    plt.savefig(pathfig, dpi=500)
    plt.close()
    return


def plot_polyfit(x_data, y_data, deg_list, x_label, y_label, title, pathfig):
    '''
    Plot polynomial models for degrees in deg_list over data.
    '''
    # create plot
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    # plot raw data points
    ax.plot(x_data, y_data, 'k.')
    colors = plt.cm.rainbow(np.linspace(0,1,len(deg_list)))
    for deg in deg_list:
        # fit polynomial to data and calculated rmse
        y_model, rmse = polyfit_rmse(x_train=x_data,
                                          y_train=y_data,
                                          x_test=x_data,
                                          y_test=y_data,
                                          deg=deg)
        # plot model with rmse in legend    
        ax.plot(x_data, y_model,
                '-', color=colors[deg], 
                label=f"Deg={deg}, RMSE={np.round(rmse, 3)}")
    # figure formatting
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc='lower left', title='Legend')
    ax.grid(True)
    fig.suptitle(title)
    plt.savefig(pathfig, dpi=500)
    plt.close()
    return


def plot_df_as_table(dataframe, title, pathfig):
    '''
    Plot a pandas dataframe as a table.
    '''
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=dataframe.values,
                     cellLoc='center', 
                     rowLabels=dataframe.index,
                     rowColours=np.full(len(dataframe.index), 'lavender'),
                     rowLoc='center',
                     colLabels=dataframe.columns, 
                     colColours=np.full(len(dataframe.columns), 'lavender'),
                     loc='center')
    table.set_fontsize(14)
    table.scale(1, 1.5)
    fig.tight_layout()
    ax.set_title(title, fontsize=14)
    plt.savefig(pathfig, dpi=500, bbox_inches='tight')
    plt.close()
    return


def plot_relative_density_hist(data, bins, ax, color, title=None):
    '''
    Plot relative density histogram on specified axis for given dataset. 
    Normalizes histogram by dividing counts by total histogram area. 
    INPUTS
        data    : np array, 1xn : Array of data values.
        color   : str           : Color to plot histogram bars.
        ax      : pyplot handle : Handle to axis to plot histogram on (e.g. subplot axis).
        title   : str           : Optional title for plot. Defaults to "Relative Density Histogram".
    RETURNS
        Relative density histogram will be plotted on given axis.
    '''
    # remove nans and flatten array
    data = data[~np.isnan(data)]

    # calculate histogram
    hist, bin_centers, width = calc_hist(data, bins)
    # normalize histogram
    hist_norm = hist / np.sum(hist * width)

    # plot on provided axis
    ax.bar(bin_centers, hist_norm, width=width, color=color)
    # label subfigure and y-axis
    #ax.set_ylabel('Density')
    if title != None:
        ax.set_title(title)
    #else:
        #ax.set_title('Relative Density Histogram')
    return