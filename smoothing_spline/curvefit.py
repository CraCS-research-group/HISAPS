# -*- coding: utf-8 -*-
"""
############################################################################
#  This Python file is part of HISAPS for fitting smoothing splines with   #
#  penalties on high order derivatives                                     #
#                                                                          #
#  The code was developed at Department of Materials and Production at     #
#  Aalborg University by  P.H. Broberg, E. Lindgaard, A.M. Olesen,         #
#  S.M. Jensen, N.K.K. Stagsted, R.L. Bjerg, R. Grosselle, I. Urcelay Oca, # 
#  B.L.V. Bak                                                              #
#                                                                          #
#  A github repository, with the most up to date version of the code,      #
#  can be found here:                                                      #
#     https://github.com/CraCS-research-group/HISAPS                       #
#                                                                          #
#  The code is open source and intended for educational and scientific     #
#  purposes. If you use this script in your research, the developers would #
#  be grateful if you could cite the paper.                                #
#                                                                          #
#  Disclaimer:                                                             #
#  The authors reserve all rights but do not guarantee that the code is    #
#  free from errors. Furthermore, the authors shall not be liable in any   #
#  event caused by the use of the program.                                 #
############################################################################

This module contains functionalities for fitting a smoothing spline to a set of
xy-data. 

References
-----------
[1] Piegl, L and Tiller, W. Monographs in Visual Communication, 1997

[2] Wand, MP and Ormerod, JT. On semiparametric regression with O'sullivan 
    penalized splines, Australian and New Zealand Journal of Statistics 
    (50), 2008, 179-198.

[3] Hastie, T, Tibshirani, R and Friedman, J. The Elements of Statistical 
    Learning: Data Mining, Inference, and Prediction. Springer Series in 
    Statistics Vol. 27, 2009
    
[4] Eugene Prilepin and Shamus Husheer, Csaps - cubic spline approximation
    (smoothing).URL - https://github.com/espdev/csaps/pull/47


"""
import smoothing_spline
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la
from scipy.optimize import minimize_scalar

from scipy.sparse import identity
from scipy.sparse.linalg import MatrixRankWarning  # The custom warning class
import warnings


plot_par = [] # Initialize plot parameters list for plotting

def b_matrix(x, t, p):
    """
    Function for computing the B-matrix. 

    Parameters
    ----------
    x : float
        Parameter.
    t : list
        Knot vector.
    p : int
        Order of the spline.

    Returns
    -------
    B : numpy array
        B-matrix.

    """
    B = np.zeros([x.size, len(t)-p-1])
    for i in range(x.size): 
        span = smoothing_spline.bsplines.find_span(x[i], p, t)
        for j in range(span-p,span+1):
            B[i,j] = smoothing_spline.bsplines.basis_fun(p, t, j, x[i], 0)
    return B

def omega(t, m):
    """
    Function for computing the omega-matrix. 
    The numerical integration is based on [2].
    
    Parameters
    ----------
    t : list
        knot vector.
    m : TYPE
        Order of the smoothing spline.

    Returns
    -------
    omega : numpy array
        Omega-matrix.

    """
    x_tilde = np.zeros((2*m-1)*(len(t)-1))
    if m == 1:
        x_tilde = np.array(t)
        wts     = np.repeat(np.diff(t),1) * np.tile((np.array([1])), len(t)-1)
    elif m == 2:
        for i in range(len(t)-1):
            x_tilde[3*i:3*i+3] = np.array([t[i], (t[i]+t[i+1])/2, t[i+1]])
        wts = np.repeat(np.diff(t),3) *                                        \
              np.tile((np.array([1,4,1]))/6, len(t)-1)
    elif m == 3:
        for i in range(len(t)-1):
            x_tilde[5*i:5*i+5] = np.array([t[i], (3*t[i]+t[i+1])/4,            \
                                  (t[i]+t[i+1])/2, (t[i]+3*t[i+1])/4, t[i+1]])
        wts = np.repeat(np.diff(t),5) *                                        \
              np.tile((np.array([14,64,8*3,64,14]))/(45*4), len(t)-1)
    elif m == 4:
        for i in range(len(t)-1):
            x_tilde[7*i:7*i+7] = np.array([t[i], (5*t[i]+t[i+1])/6,            \
                                 (2*t[i]+t[i+1])/3, (t[i]+t[i+1])/2,           \
                                 (t[i]+2*t[i+1])/3, (t[i]+5*t[i+1])/6, t[i+1]])
        wts = np.repeat(np.diff(t),7) *                                        \
              np.tile((np.array([41,216,27,272,27,216,41]))/(140*6), len(t)-1)
    else:
        print('Invalid order of smoothing spline. m should be between 1 and 4')
    
    Bdd = np.zeros([(2*m-1)*(len(t)-1), len(t)-2*m])
    
    for i in range(Bdd.shape[0]): # Make this banded at some point
        for j in range(Bdd.shape[1]):
            Bdd[i,j] = smoothing_spline.bsplines.basis_fun((2*m-1), t, j, x_tilde[i], m)
    omega = np.transpose(Bdd) @ np.diag(wts) @ Bdd
    return omega

def add_constraints(M, b, t, p, constraints):
    """
    Function for adding Lagrange multipliers to the linear equations. 

    Parameters
    ----------
    M : numpy array
        Matrix for linear equation.
    b : numpy vecotr
        Vector for linear equation.
    t : list
        Knot vector.
    p : int
        Order of the spline.
    constraints : list of dictionaries
        Contains the equality constraints to be added to the equations.

    Returns
    -------
    A : numpy array
        Updated matrix for linear equation.
    bb : numpy vector
        Updated vector for the linear equation.

    """
    if len(constraints) == 0:
        return M,b
    R = sp.lil_matrix((len(constraints), M.shape[0]))
    c = np.zeros(len(constraints))
    for i in range(len(constraints)):
        span = smoothing_spline.bsplines.find_span(constraints[i]['x'], p, t)
        # Calculate R matrix
        for j in range(span-p,span+1):
            R[i,j] = smoothing_spline.bsplines.basis_fun(p,t,j,constraints[i]['x'], constraints[i]['der'])
        # Calculate c matrix 
        c[i] = constraints[i]['f(x)']
    
    zero = sp.lil_matrix((R.shape[0], R.shape[0]))
    A1   = sp.hstack([M, np.transpose(R)], format = 'csr')
    A2   = sp.hstack([R, zero], format = 'csr')
    A    = sp.vstack([A1, A2])
    bb   = np.hstack((b,c))
    return A, bb

def cross_validation(lam, y, B, omega_jk):
    """
    Calculates the CV value for a given smoothing parameter lam, using leave-
    one-out cross validation. Based on Eq. (5.26) and (5.27) in [3]

    Parameters
    ----------
    lam : float
        Non-normalised smoothing parameter.
    y : numpy vector
        y-values
    B : numpy array
        B-matrix.
    omega_jk : nump array
        Omega-matrix.

    Returns
    -------
    float
        CV value at the given lam.

    """
    smoother = B @ np.linalg.inv(np.transpose(B) @ B +                         \
                                 lam * omega_jk) @ np.transpose(B)  
    f_lam = smoother @ y
    cv = 0.0
    for i in range(y.size):
        cv += ((y[i]-f_lam[i])/(1-smoother[i,i]))**2 
    return 1/y.size*cv 

def normalize_smooth(x, smooth, m):
    """
    Normalise the smoothing parameter using a modified approach of [4]

    Parameters
    ----------
    x : numpy array
        Vector of x-values.
    smooth : float
        Non-normalised smoothing parameter.
    m : int
        Order of the smoothing spline.

    Returns
    -------
    k : int
        Factor for normalizing the smoothing parameter.

    """
    span    = np.ptp(x)
    factor  = 2*m-1
    w       = 1/x.size * np.ones(x.size)
    eff_x   = 1 + (span ** 2) / np.sum(np.diff(x) ** 2)
    eff_w   = np.sum(w) ** 2 / np.sum(w ** 2)
    k       = factor**m * (span ** factor) * (x.size ** (-2*(factor/3))) *     \
              (eff_x ** -(0.5*(factor/3)))  * (eff_w ** -(0.5*(factor/3))) 
    return k

def cv_fun(p, x, y, B, omega_jk, sp_order):
    k = normalize_smooth(x, p, sp_order)
    lam_cv = k*(1-p)/p
    CV = cross_validation(lam_cv, y, B, omega_jk)
    return CV

def auto_smooth(x, y, B, omega_jk, sp_order):
    """
    Function to automatically chose the smoothing parameter based on 
    minimisation of the scalar CV function.

    Parameters
    ----------
    x : array
        x values of the deflection curve.
    y : array
        y values of the deflection curve.
    B : array
        B-matrix from the smoothing spline.
    omega_jk : array
        Omega matrix from the smoothing spline.
    sp_order : int
        order of the smoothing spline.

    Returns
    -------
    spar : float
        Automatically chosen smoothing parameter.

    """
    res = minimize_scalar(cv_fun, args=(x,y,B,omega_jk, sp_order),             \
                          bounds =(0,1), method='bounded')
    spar = res.x
    return spar

def cv_plot(x, y, B, omega_jk, par, spar_plot):
    """
    Function for plotting the CV values

    Parameters
    ----------
    x : array
        x values of the deflection curve.
    y : array
        y values of the deflection curve.
    B : array
        B matrix from the smoothing spline
    omega_jk : array
        Omega matrix from the smoothing spline.
    par : dict
        parameters used for the analysis.
    spar_plot : float
        Smoothing parameter used for the analysis.

    """
    n_cv = 100 
    p_cv = np.linspace(0.01, 0.99999,n_cv) 
    lam_cv = np.zeros(n_cv)
    for i in range(n_cv):
        k = normalize_smooth(x, p_cv[i], par['m'])
        lam_cv[i] = k*(1-p_cv[i])/p_cv[i]
        cv = np.zeros(n_cv)
    for i in range(n_cv):
        cv[i] = cross_validation(lam_cv[i], y, B, omega_jk)
    plt.figure('CV PLOT')
    plt.plot(p_cv,cv, label='CV values')
    plt.axvline(spar_plot, color='r', label='Chosen CV value')
    plt.title('CV')
    plt.xlabel('p'); plt.ylabel('CV')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    if 'results_dir' in par:
        plt.savefig(par['results_dir']+'/CV_plot.pdf')
    return

def solve_linear_system_with_regularization(A, bb):
    regularization_range = np.logspace(-20, 15, 36)  # Regularization parameter range
    best_residual_norm = np.inf
    best_solution = None
    best_reg_param = None
    consecutive_error_increases_regularized_solution = 0
    with warnings.catch_warnings():
        # Convert warnings to exceptions
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            #parameters = la.spsolve(sp.csr_matrix(A),bb)
            parameters = la.spsolve(sp.csr_matrix(A), bb, permc_spec='NATURAL') # NATURAL selected to avoid errors
            return parameters
        # Maybe make new except with natural permc
        except MatrixRankWarning as e:
            # If the matrix is singular, apply Tikhonov regularization
            if 'singular' in str(e):
                print("Matrix is singular. Applying Tikhonov regularization in an attempt to solve anyways...")
                for reg_param in regularization_range:
                    try:
                        print('trying with regularization parameter: ' + str(reg_param))
                        # Try solving the regularized system
                        
                        # Create a sparse identity matrix with the same shape as A
                        identity_matrix = identity(A.shape[0], format='csr')
                        A_reg = A + reg_param * identity_matrix
                        
                        x_reg = la.spsolve(A_reg, bb)  # Regularized solution
                        
                        # Calculate the dot product result
                        dot_product_result = np.dot(A.toarray(), x_reg)
                        
                        # Calculate the residual norm (error) of the solution
                        residual_norm = np.linalg.norm(dot_product_result - bb)    
                        # Check if the residual norm is increasing
                        if residual_norm > best_residual_norm:
                            consecutive_error_increases_regularized_solution += 1
                        else:
                            consecutive_error_increases_regularized_solution = 0
                        
                        # Update best solution if the residual norm is smaller
                        if residual_norm < best_residual_norm:
                            best_solution = x_reg
                            best_reg_param = reg_param
                            best_residual_norm = residual_norm
    
                    except MatrixRankWarning:
                        # Skip if regularization causes singularity
                        continue
                    # Break out of the loop if residual_norm is increasing three times
                    if consecutive_error_increases_regularized_solution >= 3:
                        print('The error from adding more regularization seems to be increasing for the last 3 increases of the regularization parameter')
                        break
    
                if best_solution is None:
                    raise Warning("Unable to find a suitable regularization parameter.")
    
                print("The system of equations has been solved with a regularization parameter of:", best_reg_param)
                return best_solution
            
            else:
                # If it's another type of error, raise it
                raise e

def spline_interpolation(B, Bt, y, t, p, con):
    ######## Special condition if p = 1 ################
    if B.shape[0] > B.shape[1]: # Inserted 21 march 2022
        M = Bt @ B
        M = sp.lil_matrix(M)
        b = np.matmul(Bt, y)
        A,bb = add_constraints(M, b, t, p, con)
        parameters = la.spsolve(A,bb)
        parameters = parameters if len(con) == 0                           \
                                else parameters[:-len(con)]
        spf = (np.array(t), np.array(parameters), p) 
        return spf
    # Solve the underdetermined set of equations using least norm
    A_dagger = Bt @ np.linalg.inv(B @ Bt)
    parameters = A_dagger @ y
    spf = (np.array(t), np.array(parameters), p) 
    return spf

    

def spline_fit(x, y, par):
    print('Fitting the smoothing spline..')
    ########## Compute spline values #############
    # fit the smoothing spline
    #spf = run_curvefit(x, y, par) # spf contains spline parameters. Maybe change to smoothing spline
    segments = 0
    thin = False
    spar = par['p']
    
    m = par['m']
    p = 2*m-1 # Spline degree
    t = smoothing_spline.bsplines.knots(list(x),p, thin, segments) # Change to numpy array at a point
    
    ######## Compute B and Omega matrices #############
    # Create b-matrix 
    B = b_matrix(x, t, p)
    Bt = np.transpose(B)

    # Calculate Omega matrix
    omega_jk = omega(t, m)
    
    ######## Special condition if p = 1 ################
    if spar == 1.0: return spline_interpolation(B, Bt, y, t, p, par['con'])
    
    ######## Compute smoothing (if p='auto') #############
    if spar == 'auto':
        print('Automatic smoothing parameter selection started:')
        spar = auto_smooth(x, y, B, omega_jk, m)
        print('Automatic smoothing parameter is '+str(spar))

    #### Normalise p and compute lambda ####
    k = normalize_smooth(x, spar, m)
    lam = k*(1-spar)/spar
   
    ### Add weights - functionality to be added to the script ####
    w = np.logspace(1, 1, x.size)
    w = w / (np.sum(w)/w.size)   
    W = np.diag(w)
    
    if par['plot']: cv_plot(x, y, B, omega_jk, par, spar)
    
    # The variable con_order gives the order of the spline that constraints are
    # added to
    smallNumber = 1e-9 
    con = par['con']
    if 'ineq_con' in par: 
        ineq_con = par['ineq_con']
    else: 
        ineq_con = ()
    par_plot_old = par['plot'] 
    par['plot'] = False  
    
    number_of_violatednequality_constraints = 1 # a number larger than 0 to start the while loop
    while number_of_violatednequality_constraints > 0:
        number_of_violatednequality_constraints = 0
        
        # update parameters
        par['con'] = con
        
        #### Add equality constraints #####
        M = Bt @ W @ B + lam*omega_jk # w inserted
        M = sp.lil_matrix(M)
        b = np.matmul(Bt, y)
        A,bb = add_constraints(M, b, t, p, con)

        #### Solve linear equation ####
        # Using cholesky
        parameters = solve_linear_system_with_regularization(A, bb)
        parameters = parameters if len(con) == 0 else parameters[:-len(con)]
        
        # Assemble spline parameters
        spf = (np.array(t), np.array(parameters), p) 
        
        # Compute relevant derivatives
        y_values = []
        for idx in range(len(ineq_con)):
            y_values.append(np.array([smoothing_spline.bsplines.spline_calc(a, spf, der=ineq_con[idx]['der']) for a in x]))

        ########## Check if constraints are satisfied #############
        indices_above_treshold = []
        for idx in range(len(ineq_con)):
            if ineq_con[idx]['inequality'] == 'greater_than':
                indices_above_treshold.append(np.where(y_values[idx] < ineq_con[idx]['treshold'])[0])
            elif ineq_con[idx]['inequality'] == 'less_than':
                indices_above_treshold.append(np.where(y_values[idx] > ineq_con[idx]['treshold'])[0])
            else:
                raise ValueError("Incorrect inequality assigned. Must be either 'greater_than' or 'less_than'")

        ########## Add constraints #############
        con_added = 0
        for idx in range(len(ineq_con)):
            if ineq_con[idx]['inequality'] == 'greater_than' and len(indices_above_treshold[idx] > 0) and con_added == 0:
                con = con + ({'x':x[indices_above_treshold[idx][0]], 'f(x)': ineq_con[idx]['treshold']  + smallNumber, 'der':ineq_con[idx]['der']}, ) # this creates a new tuble which is not speed optimal
                print('Adding constraint on deriative no: ' + str(ineq_con[idx]['der']) + '. Curent number of constraints: ' + str(len(con)))
                number_of_violatednequality_constraints += len(indices_above_treshold[idx])
                con_added = 1
            elif ineq_con[idx]['inequality'] == 'less_than' and len(indices_above_treshold[idx] > 0) and con_added == 0:
                con = con + ({'x':x[indices_above_treshold[idx][0]], 'f(x)': ineq_con[idx]['treshold']  - smallNumber, 'der':ineq_con[idx]['der']}, ) # this creates a new tuble which is not speed optimal
                print('Adding constraint on deriative no: ' + str(ineq_con[idx]['der']) + '. Curent number of constraints: ' + str(len(con)))
                number_of_violatednequality_constraints += len(indices_above_treshold[idx])
                con_added = 1

        if len(con) > 50:  # Added by PHB to stop the loop
            print('Maximum number of constraints exceeded')
            break
        
    print('Total number of constraints: ' + str(len(con)))
    
    par['plot'] = par_plot_old 
    
    if par['plot'] == True:
        fig,axes = smoothing_spline.utility.subplot_creator('Curvefit',2,3)
        
        x_sp = np.linspace(0 , np.max(x), 100)
        y_sp = np.array([smoothing_spline.bsplines.spline_calc(a, spf) for a in x_sp]) 
     
        plot_par.append( {"Title": 'Fabric deflection (Post erosion)',         \
                          "Order": 0, "Type": 'scatter', "Data": (x,y),        \
                              'Label':'Data points'} )
        plot_par.append( {"Title": 'Fabric deflection (Post erosion)',         \
                          "Order": 0, "Type": 'xy', "Data": (x_sp,y_sp),       \
                              'Label': 'Spline'} )
     
        plot_par.append( {"Title": 'Spline', "Order": 1, "Type": 'xy',         \
                          "Data": (x_sp,y_sp)} )
    
        # Spline derivatives. Plots to used to check if constraints are satisfid
        y_spd = np.array([smoothing_spline.bsplines.spline_calc(a, spf, der=1) for a in x_sp]) 
        plot_par.append( {"Title": 'Spline derivative', "Order": 2,            \
                          "Type": 'xy', "Data": (x_sp,y_spd)} )
    
        y_spc   = np.array([smoothing_spline.bsplines.spline_calc(a, spf, der = 2) for a in x_sp]) 
        plot_par.append( {"Title": 'Spline curvature' , "Order": 3,            \
                          "Type": 'xy', "Data": (x_sp,y_spc)} )
    
        y_spcc   = np.array([smoothing_spline.bsplines.spline_calc(a, spf, der = 3) for a in x_sp]) 
        plot_par.append( {"Title": 'Spline 3rd derivative' , "Order": 4,       \
                          "Type": 'xy', "Data": (x_sp,y_spcc)} )
        
        y_spccc   = np.array([smoothing_spline.bsplines.spline_calc(a, spf, der = 4) for a in x_sp]) 
        plot_par.append( {"Title": 'Spline 4th derivative' , "Order": 5,       \
                          "Type": 'xy', "Data": (x_sp,y_spccc)} )
        
        smoothing_spline.utility.subplot_filler(axes,plot_par)
    
    return spf