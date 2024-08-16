import smoothing_spline
import pickle
import numpy as np
from smoothing_spline.bsplines import*

### Integration test 
def test_full_integration():
    ## Test if the program gives the correct spline coefficients and knots
    
    # Load the input - x, y, par
    inp_file = open("test/x_y_par.pkl", "rb")
    inp = pickle.load(inp_file)
    inp_file.close()
    
    x = inp[0]
    y = inp[1]
    par = inp[2]
    
    # Load the baseline results for comparison
    result_file = open("test/baseline.pkl", "rb")
    result = pickle.load(result_file)
    result_file.close()
    
    # Run analysis
    par['plot'] = False
    new_result =  smoothing_spline.curvefit.spline_fit(x,y,par)
    
    # Assert results
    tol= 1e-4

    assert np.allclose(new_result[0],result[0], atol=tol) # knots
    assert np.allclose(new_result[1],result[1], atol=tol) # b-spline coefficient
    assert np.allclose(new_result[2],result[2], atol=tol) # degree of the spline

def test_full_integration_equality_constraint():
    ## Test if the program correctly applies equality constraints
    
    # Load the input - x, y, par
    inp_file = open("test/x_y_par.pkl", "rb")
    inp = pickle.load(inp_file)
    inp_file.close()
    
    x = inp[0]
    y = inp[1]
    par = inp[2]
    
    # Define constraints for testing
    par['con'] = ({'x':5.0, 'f(x)':0.0 , 'der':0 },
                  {'x':0.0, 'f(x)':0.0 , 'der':1 },
                  {'x':0.0, 'f(x)':1.0 , 'der':2 },)
    par['plot'] = False
    
    # Run analysis and compute spline values for comparison
    spf =  smoothing_spline.curvefit.spline_fit(x,y,par)
    new_results_0 = smoothing_spline.bsplines.spline_calc(5, spf, der=0)
    new_results_1 = smoothing_spline.bsplines.spline_calc(0, spf, der=1)
    new_results_2 = smoothing_spline.bsplines.spline_calc(0, spf, der=2)
    
    # Assert results
    tol= 1e-8

    assert np.allclose(new_results_0, 0, atol=tol) # knots
    assert np.allclose(new_results_1, 0, atol=tol) # knots
    assert np.allclose(new_results_2, 1, atol=tol) # knots

# Ready to test once the functionality is implemented. 
def test_full_integration_inequality_constraint():
    ## Test if the program correctly applies inequality constraints
    
    # Load the input - x, y, par
    inp_file = open("test/x_y_par.pkl", "rb")
    inp = pickle.load(inp_file)
    inp_file.close()
    
    x = inp[0]
    y = inp[1]
    par = inp[2]
    
    # Define constraints for testing
    par['ineq_con'] = ({'inequality':'less_than', 'treshold':0.8, 'der':0 }, ) 
    par['plot'] = False
    
    # Run analysis and compute spline values for comparison
    spf =  smoothing_spline.curvefit.spline_fit(x,y,par)
    new_results = np.array([smoothing_spline.bsplines.spline_calc(a, spf) for a in x]) 

    # Assert results
    assert np.all(new_results < 0.8) # knot

### Unit tests 

def test_auto_smooth():
    ## Test if the program computes the same smoothing parameter as baseline 
    
    # Load the input - x, y, par
    inp_file = open("test/x_y_par.pkl", "rb")
    inp = pickle.load(inp_file)
    inp_file.close()
    
    x = inp[0]
    y = inp[1]
    par = inp[2]
    
    # Compute the smoothing parameter with the automatic method
    p = 2*par['m']-1 # Spline degree
    t = knots(list(x),p, 0, 0) # Change to numpy array at a point

    #### Create b-matrix ####
    B = smoothing_spline.curvefit.b_matrix(x, t, p)
    Bt = np.transpose(B)

    #### Calculate Omega matrix ####
    omega_jk = smoothing_spline.curvefit.omega(t, par['m'])
    
    spar = smoothing_spline.curvefit.auto_smooth(x, y, B, omega_jk, par['m'])
    
    # Assert result
    tol= 1e-8

    assert np.allclose(spar, 0.0004531038537848227, atol=tol) # knots

def test_b_matrixl():
    ## Test if the program correctly computes the B-spline values in the B-matrix
    x = [0, 1, 2, 3]
    p = 2
    t = smoothing_spline.curvefit.knots(x, 2)
    
    test_B = np.array([[1,0,0,0,0], [0,0.5,0.5,0,0], [0,0,0.5,0.5,0], [0,0,0,0,1]])
    
    assert (test_B == smoothing_spline.curvefit.b_matrix(np.array(x), t, p)).all()