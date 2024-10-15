#!/usr/bin/python3
import numpy as np
import pandas as pd

def polyfit_rmse(x_train, y_train, x_test, y_test, deg):
    # find polynomial
    p = np.polyfit(x_train, y_train, deg=deg)
    # fit polynomial to data
    y_model = np.polyval(p, x_test)
    # calculate root-mean-square error
    rmse = root_mean_square_error(y_model, y_test)
    return y_model, rmse



def root_mean_square_error(model, measurements):
    assert len(model) == len(measurements)
    N = len(measurements)
    rmse = np.sqrt(1/N * np.sum((model - measurements)**2))
    return rmse

def monte_carlo_polyfit(data, n_samp, n_iters, deg_list):
    #TODO FIX DOCS
    '''
    Perform Monte Carlo simulation by taking random samples of dataset.
    Fits polynomials of degrees up to deg
    INPUTS
        data        : np array, 1xn       : Dataset to perform Montel Carlo simulation on.
        n_samp      : int                 : Number of samples; sample size of simulation data.
        n_iters     : int                 : Number of iterations; number of times to run simulation.
    RETURNS
        mc_stats    : dict of list, 4x1xn : Dictionary containing mean, standard deviation, minimum, 
            and maximum values for the sample dataset at each iteration.
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

