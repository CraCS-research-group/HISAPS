# -*- coding: utf-8 -*-
"""
This script provides two examples of how to use the smoothing_spline module to 
perform curve fitting with smoothing splines on xy-data sets.

The spline fitting is carried out with the spline_fit function in the curvefit
module. 

[Mention the paper here]

Parameters that need to be specified for the spline_fit:
    ----------------------------------
    x : numpy vector
        x values for the data set
    y : numpy vector
        y values for the data set
    p : float, str
        Smoothing parameter used in the spline fit. Should be a float between 
        0 and 1. If it is specified as 'auto', automatic parameter selection
        is used
    con : dict
        Equality constraints imposed on the spline fit. 
    ineq_con : dict
        Inequality constraints imposed on the spline fit. 
    m : int, optional
        Order of the smoothing spline. The default is m = 4
    plot : bool, optional
        If set to True the script will make plots of each step to help in 
        debugging. The default is plot = False
          
@author: Peter Broberg, Esben Lindgaard, Asbjørn Olesen, Simon Jensen, Inigo
Oca, Niklas Stagsted, Riccardo Groselle, Rasmus Bjerg, Brian Bak
CraCS Research Group, Aalborg University

date: August 2024
"""

import smoothing_spline
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev

plt.close('all') # close all previous figures upon running script


###############################################################################
########################### Sine signal example ###############################
###############################################################################

###### Generate x and y data ##################################################
# In this script, random x and y values following a sine signal are generated 

# Define variables for the data points
phase_shift = 2        # Phase shift of sine signal
amplitude = 1          # Amplitude of sine signal
x_max = 10.0           # Maximum x value
n_data_points = 50     # Number of x values
noise_amplitude = .5   # Amplitude of the noise 
random_seed = 0        # Change for other realisations of random numbers

# Generate the data points
x = np.linspace(0, x_max, n_data_points)
y_sine = smoothing_spline.utility.sine_signal(x, amplitude, phase_shift)
y = smoothing_spline.utility.noise_generator(y_sine, noise_amplitude, random_seed)

###### Smoothing spline settings ##############################################
m = 4 # Order of the smoothing spline.
      # m = 2 fits a cubic spline with penalty on second derivative
      # m = 3 fits a fifth-order spline with penalty on third derivative
      # m = 4 fits a seventh-order spline with penalty on fourth derivative

p = 'auto' # Smoothing spline parameter. If set to 'auto' it will automatically
           # determine the smoothing parameter based in minimisation algorithm.

con = ()  # No equality constraints on the fit. 

# Example of equality constraint from the paper
# con = ({'x':0.0,   'f(x)':sine_signal(0, amplitude, phase_shift), 'der':0 },
#       {'x':x_max, 'f(x)':sine_signal(x_max, amplitude, phase_shift), 'der':0 },
#       {'x':0.0,   'f(x)':sine_signal(0, amplitude, phase_shift, der=1), 'der':1 },
#       {'x':x_max, 'f(x)':sine_signal(x_max, amplitude, phase_shift, der=1), 'der':1 },
#       {'x':0.0,   'f(x)':sine_signal(0, amplitude, phase_shift, der=2), 'der':2 },
#       {'x':x_max, 'f(x)':sine_signal(x_max, amplitude, phase_shift, der=2), 'der':2 },)  

ineq_con = () # No inequality constraints on the fit. 

# Example of inequality constraint. Use 'less_than' or 'greater_than'  
# ineq_con = ({'inequality':'less_than', 'treshold':0.0, 'der':1},)                 

plot = False # Set to True for generating extra plots during fitting algorithm.

# Store parameters in dictionary   
par = {'m':m, 'p':p, 'con':con, 'ineq_con':ineq_con, 'plot':plot}     

###### Run the analysis  ######################################################
spf = smoothing_spline.curvefit.spline_fit(x, y, par)

###### Plot the fit and data points ###########################################
xp = np.linspace(0,np.max(x),500)
yp_real = smoothing_spline.utility.sine_signal(xp, amplitude, phase_shift)
yp = np.array([smoothing_spline.bsplines.spline_calc(a, spf) for a in xp]) 

plt.figure(figsize=(6.4,2.4))
plt.plot(xp,yp, label = 'Spline Fit')
plt.plot(xp,yp_real, 'k--', label = 'Signal', alpha =.5)
plt.scatter(x, y, s = 7, color = 'r', label = 'Data points')
plt.legend()
plt.xlabel('x'); plt.ylabel('y')
#Added
plt.show()

# Plot of first derivative
yp1_real = smoothing_spline.utility.sine_signal(xp, amplitude, phase_shift, der=1)
yp1 = np.array([smoothing_spline.bsplines.spline_calc(a, spf, der = 1) for a in xp]) 

plt.figure(figsize=(6.4,2.4))
plt.plot(xp,yp1, label = 'Spline Fit')
plt.plot(xp,yp1_real, 'k--', label = 'Signal', alpha =.5)
plt.legend()
plt.xlabel('x'); plt.ylabel('$dy / dx $')
plt.show()

# Plot of second derivative
yp2_real =smoothing_spline.utility.sine_signal(xp, amplitude, phase_shift, der=2)
yp2 = np.array([smoothing_spline.bsplines.spline_calc(a, spf, der = 2) for a in xp]) 

plt.figure(figsize=(6.4,2.4))
plt.plot(xp,yp2, label = 'Spline fit - unconstrained')
plt.plot(xp,yp2_real, 'k--', label = 'Signal', alpha =.5)
plt.legend()
plt.xlabel('x'); plt.ylabel('$d^2y / dx^2 $')
plt.show()

###############################################################################
###################### Cantilever bending test example ########################
###############################################################################

###### Load x and y data ######################################################
# Define variables for the data points
sim_name = 'sample_data/simulator_results_nonlinear50.csv' 
noise_amp = 0.0001 # Amplitude of the random noise added 
seed_no = 0        # Change for other realisations of random numbers

# Load data
data = np.genfromtxt(sim_name, delimiter=',', skip_header=True)
x_sim = data[:,0] # Simulated deflection curve
y_sim = data[:,1] # Simulated deflection curve
curvature_comp = data[:,2] # True curvature (ground truth)

# Add noise to data 
x = x_sim
y = smoothing_spline.utility.noise_generator(y_sim, noise_amp, seed_no)

###### Smoothing spline settings ##############################################
m = 2 # Order of the smoothing spline.
      # m = 2 fits a cubic spline with penalty on second derivative
      # m = 3 fits a fifth-order spline with penalty on third derivative
      # m = 4 fits a seventh-order spline with penalty on fourth derivative

p = 'auto' # Smoothing spline parameter. If set to 'auto' it will automatically
           # determine the smoothing parameter based in minimisation algorithm.

con = ()  # No equality constraints on the fit. 

ineq_con = () # No inequality constraints on the fit.            

plot = False # Set to True for generating extra plots during fitting algorithm.

# Store parameters in dictionary   
par = {'m':m, 'p':p, 'con':con, 'ineq_con':ineq_con, 'plot':plot}  

###### Run the analysis  ######################################################
# The analysis is run for three different order of the smoothing spline
spf_2 = smoothing_spline.curvefit.spline_fit(x, y, par)

par['m'] = 3
spf_3 = smoothing_spline.curvefit.spline_fit(x, y, par)

par['m'] = 4
spf_4 = smoothing_spline.curvefit.spline_fit(x, y, par)

###### Plot the fit and data points ##########################################
x_sp = np.linspace(0 , np.max(x), 100)

y_sp2 = splev(x_sp, spf_2)
curvature_2 = smoothing_spline.utility.compute_curvature(x_sp, spf_2)

y_sp3 = splev(x_sp, spf_3)
curvature_3 = smoothing_spline.utility.compute_curvature(x_sp, spf_3)

y_sp4 = splev(x_sp, spf_4)
curvature_4 = smoothing_spline.utility.compute_curvature(x_sp, spf_4)

plt.figure()
plt.plot(x_sim, y_sim, 'k--', label = 'Real deflection curve')
plt.scatter(x, y, color='r', s = 12, label = 'Data points - Noise amplitude: 0.1 mm')
plt.plot(x_sp, y_sp2,'-', label = 'm=2 (csaps)')
plt.plot(x_sp, y_sp3,'-', label = 'm=3')
plt.plot(x_sp, y_sp4,'-', label = 'm=4')
plt.xlabel('x [m]'); plt.ylabel('y [m]')
plt.legend()
plt.show()

# Compare curvature
plt.figure()
plt.plot(x, -curvature_comp,'k--', label = 'Ground truth')
plt.plot(x_sp, -curvature_2,'-', label = 'm=2 (csaps)')
plt.plot(x_sp, -curvature_3,'-', label = 'm=3')
plt.plot(x_sp, -curvature_4,'-', label = 'm=4')
plt.xlabel('x [m]'); plt.ylabel('curvature [1/m]')
plt.legend()
plt.show()



