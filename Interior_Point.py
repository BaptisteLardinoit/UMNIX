import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.io import loadmat
from itertools import cycle
from scipy.sparse import csc_array



def interior_point2(x0, G, d, A, b, y, D, tol, iter_max, do_debug=True):
    """
    Version 2.0
    Compute a interior point method (with a Newton algorithm): Solve min Q(x) = 0.5 x.T*G*x + x.T*d subject to a.Tx = b and a.T*x>=b 

        Inputs: - G : quadratic matrix 
                - d : linear vector
                - A : Inegality constraint Matrix (mxn)
                - b : Inegality constraint vector (mx1)    
                - s : Ax - b = s, slack vector            

        Outputs: - x_star: Parsimonious vector of size P, fill with only K non-zero vectors.
    """

    ##################
    # Initialisation #
    ##################

    # Test 2

    # Get Constraint shape
    m, n = A.shape # m number of inequalities, n dimension of state space

    # Init values
    x = x0
    s = A.dot(x) - b #np.ones((m, 1))
    lambda_ = np.ones((m, 1))

    # Init lists
    x_list = [x]
    slack_list = [s]
    lambda_list =[lambda_]
    rd_list = []
    rb_list = []
    rc_list = []
    alpha_list = []
    err_list = []

    # Init parameters
    sigma = 0.3 # Choose in [0, 1]
    alpha_coef = 0.90
    alpha = 0.01
    iter = 0
    error = 10

    # Initializing constant matrices
    A_transpose = A.T
    Jacobienne = np.block([[G, -A_transpose , np.zeros((n, m))],
                        [A, np.zeros((m, m)), -np.eye(m)],
                        [np.zeros((m,n)), np.diag(s[:, 0]), np.diag(lambda_[:, 0])]])
   

    while error > tol and iter < iter_max:
        ##################
        # Compute Residus #
        ##################
        mu = (1 / m) * ((s.T).dot(lambda_)) # duality measure
        rd = G.dot(x) - A_transpose.dot(lambda_) + d # stationarité
        # Sum = np.sum(x)*np.ones(1)
        # Ax = np.block([Sum, Sum, x])
        rb = A@x - s - b
        rc = lambda_ * s - sigma * mu 
        residus = np.block([[rd],
                            [rb],
                            [rc]])

        ######################
        # Update the Jacobian #
        ######################
        Jacobienne[n+m:, n:n+m] = np.diag(s[:, 0])
        Jacobienne[n+m:, n+m:] = np.diag(lambda_[:, 0])


        ######################
        # Solve pertubed KKT #
        ######################
        delta = np.linalg.solve(Jacobienne, residus)

        delta_x = delta[:n]
        delta_lambda_ = delta[n:n+m]
        delta_s = delta[n+m:]

        
        ######################
        # Ensure feasibility #
        ######################
        # Compute alpha to ensure feasibility, all(s) > 0 and all(lambda_) > 0
        pos_idx_s = np.where(delta_s.ravel() > 0)[0] # indexes that could push y down
        if pos_idx_s.size == 0:
            alpha_max_s = np.inf
        else:
            alpha_max_s = np.min(s[pos_idx_s].ravel() / delta_s[pos_idx_s].ravel())

        pos_idx_lambda = np.where(delta_lambda_.ravel() > 0)[0]
        if pos_idx_lambda.size == 0:
            alpha_max_lambda = np.inf
        else:
            alpha_max_lambda = np.min(lambda_[pos_idx_lambda].ravel() / delta_lambda_[pos_idx_lambda].ravel())

        
        alpha_max = min(alpha_max_s, alpha_max_lambda)
        if not np.isinf(alpha_max):
            alpha = alpha_coef * alpha_max # To avoid constraint equality


        ####################
        # Update variables #
        ####################
        x       = x       - alpha * delta_x
        s       = s       - alpha * delta_s
        lambda_ = lambda_ - alpha * delta_lambda_

        iter += 1
        error = 0.5 * ((x.T)@G)@x + (d.T)@x + 0.5 * (y.T)@y

        ######################
        # Only for Debugging #
        ######################
        if do_debug:
            x_list.append(x)
            slack_list.append(s)
            lambda_list.append(lambda_)
            rd_list.append(np.linalg.norm(rd))
            rb_list.append(np.linalg.norm(rb))
            rc_list.append(np.linalg.norm(rc))
            alpha_list.append(alpha)
            #err_norm = 0.5 * np.linalg.norm(y.reshape(-1,1)-D@x, ord=2)**2
            err_list.append(error)
            print(iter)
            print(f'Sum of the coefficients:{np.round(np.sum(x),3)}')      
            print(f'Residus:{np.round(error[0][0],6)}')
            print('')

        ########################

    
    if do_debug:
        x_star = x_list[-1]
        return x_star, s, lambda_, np.array(x_list), np.array(slack_list), np.array(lambda_list), rd_list, rb_list, rc_list, alpha_list, np.array(err_list)

    return x