# HISAPS: High-Order Smoothing Spline with Automatic Parameter Selection and Shape Constraints

HISAPS is a Python software package for fitting a high-order smoothing spline to an arbitrary dataset
(*x*, *y*) with independent variable *x* and dependent variable *y*. The software allows for fitting a 
smoothing spline that penalises the 1st, 2nd, 3rd, or 4th derivative, which can be used for obtaining 
a fit with smooth derivatives. The software offers the following features:
- Fitting of a smoothing spline with a penalties on the 1st, 2nd, 3rd, or 4th derivative.
- Automatic selection of the smoothing hyperparameter through cross-validation. 
- Ability of adding equality and inequality constraints to control the fit. 

The development of HISAPS is primarily funded through research grants. To support our development efforts, please cite the following papers when using the code for research:
- P.H. Broberg, E. Lindgaard, A.M. Olesen, S.M. Jensen, N.K.K. Stagsted, R.L. Bjerg, R. Grosselle, I. Urcelay Oca, B.L.V. Bak. HISAPS: High-Order Smoothing Spline with Automatic Parameter Selection and Shape Constraints. In review
- P.H. Broberg, E. Lindgaard, C. Krogh, S.M. Jensen, G.G. Trabal, A.F.-M. Thai, B.L.V. Bak. One-click bending stiffness: Robust and reliable automatic calculation of moment-curvature relation in a cantilever bending test. Composites Part B: Engineering 260, 110763 (2023) doi:https://doi.org/10.1016/j.compositesb.2023.110763.

The code was developed at Department of Materials and Production at Aalborg University by  P.H. Broberg, E. Lindgaard, C. Krogh, S.M. Jensen, G.G. Trabal, A.F.-M Thai, A.M. Olesen, N.K.K. Stagsted, R.L. Bjerg, R. Grosselle, I. Urcelay Oca, and B.L.V. Bak. Please feel free to use and adapt the code but remember to give proper attribution.

Happy data fitting!

Packages needed (version used during development of code)
---------------
- python      (3.11.7)
- numpy       (1.26.4)
- matplotlib  (3.8.0)
- scipy       (1.11.4)

Modules
-------
The modules can be found in the `/smoothing_spline` directory. They are briefly 
described below.

#### `curvefit` 
This module contains functionalities for fitting a smoothing 
spline to the set of data points (x, y).

#### `bsplines` 
This module contains functionalities for calculating the spline basis 
functions and derivatives.

#### `utility`
This modules contains utility functions used for examples and plotting.

How to use
-------------------------------------------------------------------
The program is run with the function `smoothing_spline.curvefit.spline_fit(x, y, par)`.
The `main.py` script shows examples for defining the parameters and calling the program.  
The output of the function is tuple `spf` which contains the spline parameters and is compatible with SciPy's existing spline tools.

Input parameters
-------------------------------------------------------------------
### Required parameters:
    x : numpy vector
        x values for the data set
    y : numpy vector
        y values for the data set
    p : float, str
        Smoothing parameter used in the spline fit. Should be a float between 
        0 and 1. If it is specified as 'auto', automatic parameter selection
        is used
        
### Optional parameters (with default values):
    con : dict, optional
        Equality constraints imposed on the spline fit. 
    ineq_con : dict, optional
        Inequality constraints imposed on the spline fit. 
    m : int, optional
        Order of the smoothing spline. The default is m = 4
    plot : bool, optional
        If set to 'True' the script will make plots of each step to help in 
        debugging. The default is plot = False

Data storage
-------------------------------------------------------------------
The results from the 'smoothing_spline.curvefit.spline_fit()' function are stored in 'spf', which is a tuple with three elements: the spline knot vector, the spline coefficients, and the degree of the spline.  

Documentation
-------------------------------------------------------------------
The modules contain docstring and comments describing the functionalities, input
varibles and outputs. 


Test
-------------------------------------------------------------------
Type `!pytest` to test the code.
