#!/usr/bin/python3
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sci

import utils
import utils_plot


def main():

    # (0) load in data
    path_home = os.path.dirname(os.path.realpath(__file__))
    path_figures = os.path.join(path_home, 'figures')
    data = np.loadtxt(os.path.join(path_home, "icevelocity.txt"))

    # (1) fit polynomials deg 0-4 to data and plot
    deg_list = np.arange(0, 5, 1)
    utils_plot.plot_polyfit(data[:,0], data[:,1], deg_list, 
                    'Depth [$m$]', 'Velocity [$m/yr$]', 
                    'Glacier Velocity vs Depth, with Polynomial Models',
                    os.path.join(path_figures, "q1_polynomial_fits.png"))

    # (2) monte carlo with 90% of the data
    n_samp = int(np.ceil(0.9 * len(data)))
    mc_stats = utils.monte_carlo_polyfit(data, n_samp=n_samp, n_iters=1000, deg_list=deg_list)
    # format output dataframe for plotting
    mc_stats = format_mc_stats(mc_stats)
    # plot dataframe as table
    utils_plot.plot_df_as_table(mc_stats, 
                                title="Model Parameters for Monte Carlo Sampling of 90\\% of Data", 
                                pathfig=os.path.join(path_figures, "q2_param_table.png"))

    # (3) cross-validation with 90% of data
    rmse_vals = cross_validataion_rmse(data, 
                                       perc_train=0.9, 
                                       n_iters=1000, 
                                       deg_list=deg_list)
    # plot distribution of RMSE vals for each degree
    utils_plot.plot_hist_subplots(rmse_vals, bins=10, 
                                  title="Cross-Validataion: Distribution of RMSE Values for Polynomial Models", 
                                  x_label="RMSE", 
                                  pathfig=os.path.join(path_figures, "q3_rmse_dist.png"))
    
    # (4) use a moving window average to model the data
    win_list = [3, 10, 50]
    #TODO CHECK ME
    utils_plot.plot_moving_avg(data[:,0], data[:,1], win_list, 
                                'Depth [$m$]', 'Velocity [$m/yr$]', 
                                'Unweighted Moving Average, Velocity vs Depth',
                                os.path.join(path_figures, "q4_moving_avg.png"))
    
    # (5) use a weighted moving window average to model the data
    xlabel = "Depth [$m$]"
    ylabel = "Velocity [$m/yr$]"
    title = "Weighted Moving Average, Velocity vs Depth"
    pathfig = os.path.join(path_figures, "q5_weighted_moving_avg.png")
    x_eval = np.arange(0, np.max(data[:,0]), 0.5)
    #mov_avg_weighted = pd.DataFrame(index=x_eval,
                                    #columns=[f"Window {win}" for win in win_list])
    # initialize figure
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.plot(data[:,0], data[:,1], 'k.')
    colors = plt.cm.rainbow(np.linspace(0, 1, len(win_list)))
    # loop through different windows
    for i, win in enumerate(win_list):
        y_model = utils.moving_avg_weighted(data[:,0], data[:,1],
                                            x_eval=x_eval, win=win)
        # plot model
        ax.plot(x_eval, y_model, '-', color=colors[i], 
                label=f"{win} m window")
    # figure formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='lower left', title='Legend')
    ax.grid(True)
    fig.suptitle(title)
    plt.savefig(pathfig)
    plt.close()

    # (6) use cross-validation to determine the optimum window size that minimizes RMSE
    win_list = np.arange(start=3, stop=50, step=1)
    rmse_vals = cross_validataion_moving_window(data, 
                                                perc_train=0.9, 
                                                n_iters=1000, 
                                                win_list=win_list)
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.plot(win_list, rmse_vals.mean(axis=0), 'o-', color='k')
    ax.plot(rmse_vals.mean(axis=0).idxmin(), rmse_vals.mean(axis=0).min(), 
            'o', color='red', markersize=10)
    ax.set_xlabel("Window Length [$m$]")
    ax.set_ylabel("Mean RMSE Value")
    fig.suptitle("Mean RMSE Values for 1000 Iterations vs Weighted Moving Window Size")
    pathfig = os.path.join(path_figures, "q6_crossval_weighted_moving_avg.png")
    plt.savefig(pathfig)
    plt.close()

    # (7) use brute-force method to find optimum values of parameters A and n
    v0 = data[0,1]      # m/s
    rho = 917           # kg/m3
    grav = 9.8          # m/s2
    theta = 10          # deg slope

    A_range = np.arange(start=1e-18, stop=10e-18, step=1e-18)
    n_range = np.arange(start=2, stop=4, step=0.01)
    # initialize arrays
    rmse_bruteforce = np.zeros([len(A_range), len(n_range)])
    for i in range(len(A_range)):
        for j in range(len(n_range)):
            # find value of model at all depths for current param values
            y_model = v0 - A_range[i] * (rho * grav * np.sin(np.deg2rad(theta)))**n_range[j] * data[:,0]**(n_range[j]+1) 
            rmse_bruteforce[i, j] = utils.root_mean_square_error(model=y_model,
                                                      measurements=data[:,1])

    ## (8) plot RMSE for values of A and n as colormap
    #fig, ax = plt.subplots(1, 1, tight_layout=True)
    #im = ax.imshow(rmse, 
    #          extent=[min(n), max(n), min(A), max(A)],
    #            # set aspect to automatically adjust
    #          aspect='auto',
    #          origin='lower',
    #          cmap='bone',
    #          vmin=0, vmax=20)
    #fig.colorbar(im, label="RMSE", orientation='vertical')
    #ax.set_xlabel("Parameter $n$")
    #ax.set_ylabel("Parameter $A$")
    #fig.suptitle("Brute Force Method: Optimal Values for Flow Parameters")
    #pathfig = os.path.join(path_figures, "q8_crossval_brute_force.png")
    #plt.savefig(pathfig)
    #plt.close()

    # (9) use the gradient descent method to find optimal values of A and n
    # choose initial guess from brute force results
    init_guess = [5e-18, 3]
    result = sci.optimize.minimize(cost_function_gradient_descent, 
                                   x0=init_guess,
                                   args=(data))
    [B, n] = result.x
    # chose slightly different cost function to manage precision
    A = B / ( (rho * grav * np.sin(np.deg2rad(theta)))**n ) 

    print("(9) Gradient Descent")
    print("\tOptimal n = ", n)
    print("\tOptimal A = ", A)

    # (10) gradient descent method with cross-validation approach using n=3
    rmse_vals, A_vals = cross_validataion_gradient_descent(data, 
                                                   perc_train=0.9, 
                                                   n_iters=1000)
    # plot distribution of A and RMSE vals
    fig, ax = plt.subplots(nrows=1, ncols=2, tight_layout=True)
    utils_plot.plot_relative_density_hist(rmse_vals, bins=30, 
                                          ax=ax[0], color='blue', 
                                          title="Values of RMSE")
    utils_plot.plot_relative_density_hist(A_vals, bins=30, 
                                          ax=ax[1], color='red', 
                                          title="Values of Parameter $A$")
    fig.suptitle("Distribution of Cross-Validataion on Gradient Descent Method")
    pathfig = os.path.join(path_figures, "q10_crossval_rmse_A_dist.png")
    plt.savefig(pathfig)
    plt.close()

    # (11) plot mean optimal A with errorbars as stdev
    A_mean = np.mean(A_vals)
    A_std = np.std(A_vals)
    rmse_mean = np.mean(rmse_vals)
    rmse_std = np.std(rmse_vals)

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    im = ax.imshow(rmse_bruteforce, 
              extent=[min(n_range), max(n_range), min(A_range), max(A_range)],
                # set aspect to automatically adjust
              aspect='auto',
              origin='lower',
              cmap='bone',
              vmin=0, vmax=20)
    ax.vlines(3, ymin=A_mean-A_std, ymax=A_mean+A_std, colors='orange',
              linewidth=3)
    ax.plot(3, A_mean, 'r*', markersize=15, label="Gradient Descent, Cross-Validation")
    ax.plot(n, A, 'b*', markersize=15, label="Gradient Descent, All Data")
    fig.colorbar(im, label="RMSE", orientation='vertical')
    ax.legend(loc='upper right')
    ax.set_xlabel("Parameter $n$")
    ax.set_ylabel("Parameter $A$")
    fig.suptitle("Brute Force Method: Optimal Values for Flow Parameters")
    pathfig = os.path.join(path_figures, "q11_comp_methods.png")
    plt.savefig(pathfig)
    plt.close()

    # (12) create normal distributions from A and RMSE values
    A_norm_dist = np.random.normal(loc=A_mean, scale=A_std, size=1000)
    rmse_norm_dist = np.random.normal(loc=rmse_mean, scale=rmse_std, size=1000)

    # (13) compare actual distributions with normal distribution
    result = sci.stats.ks_2samp(A_vals, A_norm_dist)
    pval_A_dist = result.pvalue
    result = sci.stats.ks_2samp(rmse_vals, rmse_norm_dist)
    pval_rmse_dist = result.pvalue

    print("(13) 2 Sample K-S Test")
    print("\tA Distribution P-Value = ", pval_A_dist)
    print("\tRMSE Distribution P-Value = ", pval_rmse_dist)

    return




def cost_function_gradient_descent(params, data):
    B = params[0]
    n = params[1]
    y_model = data[0,1] - B * data[:,0]**(n+1)
    rmse = utils.root_mean_square_error(model=y_model,
                                        measurements=data[:,1])
    return rmse
def cost_function_n3(params, data):
    B = params
    n = 3
    y_model = data[0,1] - B * data[:,0]**(n+1)
    rmse = utils.root_mean_square_error(model=y_model,
                                        measurements=data[:,1])
    return rmse

def theoretical_model_params():
    rho = 917           # kg/m3
    grav = 9.8          # m/s2
    theta = 10          # deg slope
    return rho, grav, theta

def theoretical_model(data, A, n):
    rho, grav, theta = theoretical_model_params()
    y_model = data[0,1] - A * (rho * grav * np.sin(np.deg2rad(theta)))**n * data[:,0]**(n+1)
    return y_model


def cross_validataion_gradient_descent(data, perc_train, n_iters):
    # use a power of n=3
    n = 3
    # initialize array to store RMSE values
    rmse_vals = []
    A_vals = []
    # perform cross-validataion 
    for i in range(n_iters):
        # choose sample of training data from entire dataset
        data_train, data_test = get_train_test(data, perc_train=perc_train)

        # find optimal parameters with training data
        result = sci.optimize.minimize(cost_function_n3, 
                                       x0=1e-8, 
                                       args=(data_train))
        B = result.x[0]
        # calculate A parameter
        rho, grav, theta = theoretical_model_params()
        A = B / ( (rho * grav * np.sin(np.deg2rad(theta)))**n ) 
        A_vals.append(A)

        # fit theoretical model to test data
        y_model = theoretical_model(data_test, A, n)
        # calculate and save RMSE
        rmse = utils.root_mean_square_error(y_model, data_test[:,1])
        rmse_vals.append(rmse)
    
    rmse_vals = np.array(rmse_vals)
    A_vals = np.array(A_vals)
    rmse_vals[np.isinf(rmse_vals)] = np.nan
    A_vals[np.isinf(A_vals)] = np.nan

    return rmse_vals, A_vals



def cross_validataion_moving_window(data, perc_train, n_iters, win_list):
    '''
        rmse_vals   : np array  : Size len(deg_list) by n_iters
    '''
    # initialize array to store RMSE values for each window length
    rmse_vals = pd.DataFrame(columns=[win for win in win_list])

    for win in win_list:
        win_rmse = []
        # perform cross-validataion 
        for i in range(n_iters):
            # choose sample of training data from entire dataset
            data_train, data_test = get_train_test(data, perc_train=perc_train)
            # find and fit polynomial model
            y_model = utils.moving_avg_weighted(x_data=data_train[:,0], 
                                                y_data=data_train[:,1],
                                                x_eval=data_test[:,0],
                                                win=win)
            rmse = utils.root_mean_square_error(y_model, data_test[:,1])
            # save individual RMSE
            win_rmse.append(rmse)
        # save list of RMSE
        rmse_vals[win] = win_rmse
    return rmse_vals

def cross_validataion_rmse(data, perc_train, n_iters, deg_list):
    '''


        rmse_vals   : np array  : Size len(deg_list) by n_iters
    '''
    # initialize array to store RMSE values for each degree polyfit
    rmse_vals = pd.DataFrame(columns=[f"Degree {deg}" for deg in deg_list])

    for deg in deg_list:
        deg_rmse = []
        # perform cross-validataion 
        for i in range(n_iters):
            # choose sample of training data from entire dataset
            data_train, data_test = get_train_test(data, perc_train=perc_train)
            # find and fit polynomial model
            y_model, rmse = utils.polyfit_rmse(x_train=data_train[:,0], 
                                               y_train=data_train[:,1],
                                               x_test=data_test[:,0], 
                                               y_test=data_test[:,1],
                                               deg=deg)
            # save individual RMSE
            deg_rmse.append(rmse)
        # save list of RMSE
        rmse_vals[f"Degree {deg}"] = deg_rmse
    return rmse_vals





def get_train_test(data, perc_train):
    #TODO fix docs anc comment
    '''
    Divides data into randomly-sampled training and test subsets.
    INPUTS
        data    : np array  : 
        perc_train : float     : Between 0 and 1
    RETURNS
    '''
    n, _ = np.shape(data)

    # number of samples in training data
    n_train = int(perc_train* n)
    idx = np.arange(1, n)
    idx_train = np.random.choice(idx, size=n_train, replace=False)#, replace=True)
    data_train = data[idx_train]

    data_test = data[[i for i in range(n) if i not in idx_train]]


    return data_train, data_test


def format_mc_stats(mc_stats):
    mc_stats.index = "Degree "+mc_stats.index.astype(str)
    mc_stats[['A0 Mean', 'A1 Mean', 'A2 Mean', 'A3 Mean', 'A4 Mean']] = pd.DataFrame(mc_stats['Param Mean'].tolist(), index= mc_stats.index)
    mc_stats[['A0 Stdev','A1 Stdev', 'A2 Stdev', 'A3 Stdev', 'A4 Stdev']] = pd.DataFrame(mc_stats['Param Stdev'].tolist(), index= mc_stats.index)
    mc_stats = mc_stats.drop('Param Mean', axis=1).drop('Param Stdev', axis=1)
    mc_stats = mc_stats.astype(float).round(3)
    mc_stats = mc_stats.transpose()
    return mc_stats



if __name__ =="__main__":
    main()