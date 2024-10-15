#!/usr/bin/python3
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import utils
import utils_plot


def main(debug):

    #TODO remove debug flag
    # load in data
    path_home = os.path.dirname(os.path.realpath(__file__))
    path_figures = os.path.join(path_home, 'figures')
    data = np.loadtxt(os.path.join(path_home, "icevelocity.txt"))

    if debug == False:
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

    elif debug == True:
        #### (3) cross-validation with 90% of data
        ###rmse_vals = cross_validataion_rmse(data, 
        ###                                perc_train=0.9, 
        ###                                n_iters=1000, 
        ###                                deg_list=deg_list)
        #### plot distribution of RMSE vals for each degree

        ###pathfig=os.path.join(path_figures, "q3_rmse_dist.png")
        

        ###fig, ax = plt.subplots(2, 5, 
        ###                    tight_layout=True, sharex=True,
        ###                    #sharey=True,
        ###                    figsize=[10,3])
        ###fig.suptitle("RMSE Values for Polynomial Models using 1000 Cross-Validation Tests",
        ###            fontsize=14)

        #### create horizontal boxplot for each dataset
        ###colors = plt.cm.rainbow(np.linspace(0, 1, len(deg_list)))
        ###for i, deg in enumerate(deg_list):
        ###    axb = ax.reshape(-1)[i]
        ###    axb.boxplot(rmse_vals[deg], vert=False)
        ###    axb.set_yticks([1], [])
        ###    axb.set_title(f"Degree {deg}")

        ###    axh = ax.reshape(-1)[i+len(deg_list)]
        ###    utils_plot.plot_relative_density_hist(rmse_vals[deg], bins=10, 
        ###                                        ax=axh, color=colors[i], title=None)
        ###    axh.set_xlabel('RMSE')
        ###ax[1,0].set_ylabel('Density')
        ####ax[0,0].set_title('Depths 1\nBoxplot')

        ###plt.savefig(pathfig, dpi=500, bbox_inches='tight')
        ###plt.close()

        # (4) moving window average
        #TODO currently in points not m
        win_list = [3, 10, 50] # m
        utils_plot.plot_moving_avg(data[:,0], data[:,1], win_list, 
                                   'Depth [$m$]', 'Velocity [$m/yr$]', 
                                   'Glacier Velocity vs Depth, with Moving Average',
                                   os.path.join(path_figures, "q4_moving_avg.png"))
        

    # (6) use cross validation



    # 2024-10-03 in class
    # (7) brute force approach
    #FIXME we need to use the correct variable names according to the HW here
    ns = len(velocity)
    A0 = np.arange(20, 150, 1)
    A1 = np.arange(-1, 0.5, 0.005)
    # initialize arrays
    rmse = np.zeros([len(A0), len(A1)])
    for n in range(len(A0)):
        for n2 in range(len(A1)):
            # find value of model at all depths for current param values
            v_model = A0[n] + A1[n2]*depth
            rmse[n,n2] = utils.root_mean_square_error(model=v_model,
                                                measurements=velocity)
    # now plot as a heatmap
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.imshow(rmse, 
              extent=[min(A1), max(A1), min(A0), max(A0)],
                # set aspect to automatically adjust
              aspect='auto',
              origin='lower',
              cmap='viridis',
              vmin=0, vmax=20)
    #ax.colorbar(label='RMSE')
    ax.set_title('RMSE values for linear model')
    # plot polyfit result which should be at center of bowl
    # only did this for deg 1 in class
    #ax.plot(p[0], p[1], 'ro', linewidth=2, label='polyfit result')
    #plt.show()

    # (9) gradient descent
    from scipy.optimize import minimize
    # define objective/cost function
    def rmse_val(P):
        data = np.loadtxt(os.path.join(path_home, "icevelocity.txt"))
        z = data[:,0]
        v = data[:,1]
        vmodel = P[0]*z + P[1]
        rmse = np.sqrt(np.mean((vmodel-v)**2))
        return rmse
    # well behave func with only one min, so initial guess shouldnt matter
    # this method is much faster
    # can use timeit
    init_guess = [-1, 20]
    result = minimize(rmse_val, init_guess)
    pbest = result.x

    ax.plot(pbest[0], pbest[1], 'wx', markersize=5, label='gradient descent')

    #plt.show()



    z = depth
    v = velocity
    zsum = np.sum(z)
    z2sum = np.sum(z**2)
    M = np.array([[zsum, len(z)], [z2sum, zsum]])
    vsum = np.sum(v)
    vzsum = np.sum(z*v)
    y = np.array([vsum, vzsum])
    pbest2 = np.linalg.solve(M, y)

    ax.plot(pbest2[0], pbest2[1], 'k^', markersize=5, label='analytic')
    plt.legend()
    plt.show()
    
        
        


    return

def cross_validataion_rmse(data, perc_train, n_iters, deg_list):
    '''


        rmse_vals   : np array  : Size len(deg_list) by n_iters
    '''
    # initialize array to store RMSE values for each degree polyfit
    rmse_vals = pd.DataFrame(columns=deg_list)

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
        rmse_vals[deg] = deg_rmse
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
    main(debug=True)