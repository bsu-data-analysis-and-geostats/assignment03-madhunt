#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils

def plot_moving_avg(x_data, y_data, win_list, x_label, y_label, title, pathfig):
    '''
    '''
    # create plot
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    # plot raw data points
    ax.plot(x_data, y_data, 'k.')

    colors = plt.cm.rainbow(np.linspace(0,1,len(win_list)))
    for i, win in enumerate(win_list):
        # calculate moving avg
        y_model = pd.DataFrame(index=x_data, data=y_data)
        y_model = y_model.rolling(window=win, center=True).mean()
        #idx = [np.arange(i-win, i+win, 1)%len(y_data) for i, y in enumerate(y_data)]
        #y_model = [np.mean(y_data[i]) for i in idx]

        # plot model with rmse in legend    
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
    '''
    # create plot
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    # plot raw data points
    ax.plot(x_data, y_data, 'k.')

    colors = plt.cm.rainbow(np.linspace(0,1,len(deg_list)))
    for deg in deg_list:
        # fit polynomial to data and calculated rmse
        y_model, rmse = utils.polyfit_rmse(x_train=x_data,
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
    hist, bin_centers, width = utils.calc_hist(data, bins)
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