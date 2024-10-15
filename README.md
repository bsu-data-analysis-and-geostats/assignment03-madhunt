# GEOPH 522: HW 3
* Author: Madeline Hunt
* Date: __2024

## (A) Code Overview

### TODO 
### ***********************
    # first col is depth (z), second col is velocity (v)    
    # holes drilled every 2 m, inclinometer in each hole and freeze over, 
    #   watch column/hole deform over time
    #   can tell how velocity of glacier changes with depth
    #   fastest vel at surface, decrease towards bed; slip at the bed, so x-intercept is non-zero
    #   want to fit data to predict velocity everywhere
    #   fit polynomial velocity model v_m(z) = A0 + A1z + A2z^2 + ... + Anz^n
    #       fit different polynomial degrees


## (B) Parametric Statistical Models

1. I fit the data using polynomial models with degrees 0-4 and plotted these overlying the data on the same figure, as seen below. Visually, the degree 3 and 4 models appear to fit the data best; these models also have the lowest RMSE. However, with increasing degree of polynomial, you become more at risk of overfitting the data.

![Glacier Velocity vs Depth, with Polynomial Models](figures/q1_polynomial_fits.png)

2. Next, I used Monte Carlo sampling to take samples of 90% of the total data and fit a model to that 90%. I repeated this 1000 times for each polynomial model degree. Results for the mean and standard deviation for each model parameter are reported in the table below, along with the RMSE.

![Model Parameters for Monte Carlo Sampling of 90% of Data ](figures/q2_param_table.png)

## (C) Cross-Validation

3. 
![](figures/q3_rmse_dist.png)



## (D) Non-Parametric Statistical Models

4. 

![Moving Average](figures/q4_moving_avg.png)

5. 

![Weighted Moving Average](figures/)

6. Optimum window size


## (E) Theoretical Ice Flow Model

7. 
8. 
9.
10. 
11. 
12.
13.


