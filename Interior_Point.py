import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.io import loadmat
from itertools import cycle
from scipy.sparse import csc_array


"""
File: code_ex_UNMIX.py
Author: Luca MIMOUNI - Baptiste LARDINOIT
Date: 2025-01-27

Description: Different version of the Interior Point method algorithm for Quadratic Problems
"""

def interior_point(x0, G, d, A, b, y, D, iter_max, do_debug=False):
    """
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
    err_quadra_list = []
    err_norm_list = []

    # Init parameters
    sigma = 0.3 # Choose in [0, 1]
    alpha_coef = 0.90
    alpha = 0.01
    iter = 0
    # iter_max = params["iter_max"]

    # TODO: csc_array and csr_array
    Jacobienne = np.block([[G, -A.T , np.zeros((n, m))],
                               [A, np.zeros((m, m)), -np.eye(m)],
                               [np.zeros((m,n)), np.diag(s[:, 0]), np.diag(lambda_[:, 0])]]) #y[:, 0]
        

    while iter < iter_max:
        # Calcul des résidus
        mu = (1 / m) * ((s.T).dot(lambda_)) # duality measure
        rd = G.dot(x) - A.T.dot(lambda_) + d # stationarité
        #TODO: transpose_A = A.T -> A.T*u = u[:n-2, :n-2] + (u[n-1]+u[n])*np.ones(n-2,2) 
        rb = A.dot(x) - s - b #TODO: Sum = np.sum(x); Ax -> np.block([Sum, Sum, x])
        rc = lambda_ * s - sigma * mu 
        residus = np.block([[rd],
                            [rb],
                            [rc]])

        #TODO: Don't build each time the full Jacobienne
        # Jacobienne[n+m+1, n+m+...] =
        # Jacobienne = np.block([[G, -A.T , np.zeros((n, m))],
        #                     [A, np.zeros((m, m)), -np.eye(m)],
        #                     [np.zeros((m,n)), np.diag(s[:, 0]), np.diag(lambda_[:, 0])]]) #y[:, 0]
        
        Jacobienne[n+m:, n:n+m] = np.diag(s[:, 0])
        Jacobienne[n+m,n+m:] = np.diag(np.diag(lambda_[:, 0]))

        delta = np.linalg.solve(Jacobienne, residus) ## TODO Test with scipy sparse

        delta_x = delta[:n]
        delta_lambda_ = delta[n:n+m]
        delta_s = delta[n+m:]

        # Compute alpha to ensure feasibility, all(s) > 0 and all(lambda_) > 0
        pos_idx_s = np.where(delta_s.ravel() > 0)[0] # indeces that could push y down
        if pos_idx_s.size == 0:
            alpha_max_s = np.inf
        else:
            alpha_max_s = np.min(s[pos_idx_s].ravel() / delta_s[pos_idx_s].ravel())

        pos_idx_lambda = np.where(delta_lambda_.ravel() > 0)[0]
        if pos_idx_lambda.size == 0:
            alpha_max_lambda = np.inf
        else:
            alpha_max_lambda = np.min(lambda_[pos_idx_lambda].ravel() / delta_lambda_[pos_idx_lambda].ravel())

        

        #print(f"alpha_max_y : {alpha_max_y}; alpha_max_lambda : {alpha_max_lambda}")
        #alpha_p = min(alpha_coef * alpha_max_y, alpha_s)
        #alpha_d = min(alpha_coef * alpha_max_lambda, alpha_s)
        alpha_max = min(alpha_max_s, alpha_max_lambda)
        if not np.isinf(alpha_max):
            alpha = alpha_coef * alpha_max # To avoid constraint equality

        # print(f"{y=}")


        # Mise a jours des variables
        x       = x       - alpha * delta_x
        s       = s       - alpha * delta_s
        lambda_ = lambda_ - alpha * delta_lambda_

        # print(f"{alpha=}")
        # print(f"{A.dot(x) - b=}")
        # print(f"{y=}")

        # Debugging
        if do_debug:
            x_list.append(x)
            slack_list.append(s)
            lambda_list.append(lambda_)
            rd_list.append(np.linalg.norm(rd))
            rb_list.append(np.linalg.norm(rb))
            rc_list.append(np.linalg.norm(rc))
            alpha_list.append(alpha)

            err_quadra = 0.5 * ((x.T)@G)@x + (d.T)@x + 0.5 * (y.T)@y
            err_norm = 0.5 * np.linalg.norm(y.reshape(-1,1)-D@x, ord=2)**2
            err_quadra_list.append(err_quadra)
            err_norm_list.append(err_norm)


        # if iter % 10:
        #     print(iter)
        #     #print(f"Jacobienne.shape : {Jacobienne.shape}; residus.shape : {residus.shape}")
        #     #print(f"rd.shape : {rd.shape}, rb.shape : {rb.shape}, rc.shape : {rc.shape}")
        #     #print(f"delta_x.shape : {delta_x.shape}, delta_y.shape : {delta_y.shape}, delta_lambda_.shape : {delta_lambda_.shape}")
        x_star = x_list[-1]

        iter += 1
    
    return x_star, s, lambda_, np.array(x_list), np.array(slack_list), np.array(lambda_list), rd_list, rb_list, rc_list, alpha_list, np.array(err_quadra_list), np.array(err_norm_list)



def interior_point2(x0, G, d, A, b, A_bar, b_bar, y, D, tol, iter_max, do_debug=True):
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



def interior_point3(x0, G, d, A, b, A_bar, b_bar, y, D, tol, iter_max, do_debug=True):
    """
    Version 3.0
    Compute a interior point method (with a Newton algorithm): Solve min Q(x) = 0.5 x.T*G*x + x.T*d subject to A_bar.Tx = b_bar and A.T*x>=b 

        Inputs: - G : quadratic matrix 
                - d : linear vector
                - A : Inegality constraint Matrix (mxn)
                - b : Inegality constraint vector (mx1)    
                - A_bar : Inegality constraint Matrix (mxn)
                - b_bar : Inegality constraint vector (mx1)   
                - s : Ax - b = s, slack vector            

        Outputs: - x_star or x : Solution of the Interior Point algorithm 
    """

    ##################
    # Initialisation #
    ##################

    # Test 2

    # Get Constraint shape
    m, n = A.shape # m number of inequalities, n dimension of state space

    # Init values
    A_tild = np.block([[A], [A_bar]])
    b_tild = np.block([[b], [b_bar]])

    x = x0
    #s_bar = A_bar@x - b_bar
    s = A@x - b #np.ones((m, 1))
    mu2 = 1
    lambda_ = np.ones((m, 1))

    if do_debug:
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

    # Jacobienne = np.block([[G, -A_transpose , -A_bar.T, np.zeros((n, m))],
    #                 [A, np.zeros((m, m)), np.zeros((m, 1)), -np.eye(m,n)],
    #                 [A_bar, np.zeros((1, m)), np.zeros((1, 1)), np.zeros((1, n))],
    #                 [np.zeros((m,n)), np.zeros((m,n)), np.diag(s[:, 0]), np.diag(lambda_[:, 0])]])

    S = np.block([np.diag(s[:, 0]),np.zeros((n,1))])
    Lambda = np.diag(lambda_[:, 0])

    Jacobienne = np.block([
                [G, -A_tild.T, np.zeros((n, m))],
                [A_tild, np.zeros((m+1, m+1)), -np.eye(m+1,n)], #np.block([[-np.eye(m,n)],[np.zeros((1, n))]])
                [np.zeros((m,n)), S, Lambda]
                ])
   

    while error > tol and iter < iter_max:
        ##################
        # Compute Residus #
        ##################
        mu = (1 / m) * ((s.T)@lambda_) # duality measure
        rd = G@x - A.T@lambda_ + d # stationarité
        # Sum = np.sum(x)*np.ones(1)
        # Ax = np.block([Sum, Sum, x])
        # rb = A@x - s - b
        # rb_bar = A_bar@x - b_bar
        rb_tild = A_tild@x - b_tild
        rc = lambda_ * s - sigma * mu 
        residus = np.block([[rd],
                            [rb_tild],
                            [rc]])

        #######################
        # Update the Jacobian #
        #######################
        Jacobienne[n+m+1:, n:n+n+1] = np.block([np.diag(s[:, 0]),np.zeros((n,1))])
        Jacobienne[n+m+1:, n+n+1:] = np.diag(lambda_[:, 0])


        ######################
        # Solve pertubed KKT #
        ######################
        delta = np.linalg.solve(Jacobienne, residus)

        delta_x = delta[:n]
        delta_lambda_ = delta[n:n+m]
        delta_mu2 = delta[n+m:n+m+1]
        delta_s = delta[n+m+1:]

        
        ######################
        # Ensure feasibility #
        ######################
        # Compute alpha to ensure feasibility, all(s) > 0 and all(lambda_) > 0

        # Slack
        pos_idx_s = np.where(delta_s.ravel() > 0)[0] # indexes that could push y down
        if pos_idx_s.size == 0:
            alpha_max_s = np.inf
        else:
            alpha_max_s = np.min(s[pos_idx_s].ravel() / delta_s[pos_idx_s].ravel())

        # Lagrange multiplier lambda
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
        mu2     = mu2     - alpha * delta_mu2
        lambda_ = lambda_ - alpha * delta_lambda_

        iter += 1
        error = 0.5 * ((x.T)@G)@x + (d.T)@x + 0.5 * (y.T)@y

        ######################
        # Only for Debugging #
        ######################
        if do_debug:
            print(iter)
            print(f'mu2={mu2}')
            x_list.append(x)
            slack_list.append(s)
            lambda_list.append(lambda_)
            rd_list.append(np.linalg.norm(rd))
            rb_list.append(np.linalg.norm(rb_tild))
            rc_list.append(np.linalg.norm(rc))
            alpha_list.append(alpha)
            #err_norm = 0.5 * np.linalg.norm(y.reshape(-1,1)-D@x, ord=2)**2
            err_list.append(error)
            print(f'Sum of the coefficients: {np.sum(x)}')      
            print(f'Residus:{np.round(error[0][0],6)}')
            print('')

        ########################

    
    if do_debug:
        x_star = x_list[-1]
        return x_star, s, lambda_, np.array(x_list), np.array(slack_list), np.array(lambda_list), rd_list, rb_list, rc_list, alpha_list, np.array(err_list)

    return x


from scipy.sparse import csr_matrix, bmat, eye, block_diag, diags
from scipy.sparse.linalg import spsolve

def interior_point4(x0, G, d, A, b, A_bar, b_bar, y, D, tol, iter_max, do_debug=True):
    """
    Version 4.0
    Compute a interior point method (with a Newton algorithm): Solve min Q(x) = 0.5 x.T*G*x + x.T*d subject to A_bar.Tx = b_bar and A.T*x>=b 
    using Sparsity properties of the Jacobian.
        Inputs: - G : quadratic matrix 
                - d : linear vector
                - A : Inegality constraint Matrix (mxn)
                - b : Inegality constraint vector (mx1)    
                - A_bar : Inegality constraint Matrix (mxn)
                - b_bar : Inegality constraint vector (mx1)   
                - s : Ax - b = s, slack vector            

        Outputs: - x_star or x : Solution of the Interior Point algorithm 
    """

    ##################
    # Initialisation #
    ##################

    # Test 2

    # Get Constraint shape
    m, n = A.shape # m number of inequalities, n dimension of state space

    # Init values
    A_tild = np.block([[A], [A_bar]])
    b_tild = np.block([[b], [b_bar]])

    # Use advantage of the sparsity property
    A_tild = csr_matrix(A_tild)
    b_tild =csr_matrix(b_tild)
    zeros_nn = csr_matrix((n, n))
    zeros_n1 = csr_matrix((n, 1))
    zeros_1n = csr_matrix((1, n))
    zeros_nplus1_nplus1 = csr_matrix((n + 1, n + 1))
    zeros_nplus1_n = csr_matrix((n + 1, n))

    x = x0
    #s_bar = A_bar@x - b_bar
    s = A@x - b #np.ones((m, 1))
    mu2 = 1
    lambda_ = np.ones((m, 1))

    if do_debug:
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

    S = bmat([[diags(s.flatten()), zeros_n1]])
    Lambda = diags(lambda_.flatten())

    Jacobienne = bmat([
                [G, -A_tild.T, zeros_nn],
                [A_tild, zeros_nplus1_nplus1, -eye(m+1,n)],
                [zeros_nn, S, Lambda]
                ])
   

    while error > tol and iter < iter_max:
        ##################
        # Compute Residus #
        ##################
        mu = (1 / m) * ((s.T)@lambda_) # duality measure
        rd = G@x - A.T@lambda_ + d # stationarité
        # Sum = np.sum(x)*np.ones(1)
        # Ax = np.block([Sum, Sum, x])
        # rb = A@x - s - b
        # rb_bar = A_bar@x - b_bar
        rb_tild = A_tild@x - b_tild
        rc = lambda_ * s - sigma * mu 
        residus = np.block([[rd],
                            [rb_tild],
                            [rc]])

        #######################
        # Update the Jacobian #
        #######################
        #Jacobienne[n+m+1:, n:n+n+1] = bmat([[diags(s.flatten()), zeros_n1]])
        #Jacobienne[n+m+1:, n+n+1:] = diags(lambda_.flatten())
        
        S = bmat([[diags(s.flatten()), zeros_n1]])
        Lambda = diags(lambda_.flatten())

        Jacobienne = bmat([
                    [G, -A_tild.T, zeros_nn],
                    [A_tild, zeros_nplus1_nplus1, -eye(m+1,n)],
                    [zeros_nn, S, Lambda]
                    ])

        ######################
        # Solve pertubed KKT #
        ######################
        delta = spsolve(Jacobienne, residus).reshape(-1,1)

        delta_x = delta[:n]
        delta_lambda_ = delta[n:n+m]
        delta_mu2 = delta[n+m:n+m+1]
        delta_s = delta[n+m+1:]

        
        ######################
        # Ensure feasibility #
        ######################
        # Compute alpha to ensure feasibility, all(s) > 0 and all(lambda_) > 0

        # Slack
        pos_idx_s = np.where(delta_s.ravel() > 0)[0] # indexes that could push y down
        if pos_idx_s.size == 0:
            alpha_max_s = np.inf
        else:
            alpha_max_s = np.min(s[pos_idx_s].ravel() / delta_s[pos_idx_s].ravel())

        # Lagrange multiplier lambda
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
        mu2     = mu2     - alpha * delta_mu2
        lambda_ = lambda_ - alpha * delta_lambda_

        iter += 1
        error = 0.5 * ((x.T)@G)@x + (d.T)@x + 0.5 * (y.T)@y

        ######################
        # Only for Debugging #
        ######################
        if do_debug:
            print(iter)
            print(f'mu2={mu2}')
            x_list.append(x)
            slack_list.append(s)
            lambda_list.append(lambda_)
            rd_list.append(np.linalg.norm(rd))
            rb_list.append(np.linalg.norm(rb_tild))
            rc_list.append(np.linalg.norm(rc))
            alpha_list.append(alpha)
            #err_norm = 0.5 * np.linalg.norm(y.reshape(-1,1)-D@x, ord=2)**2
            err_list.append(error)
            print(f'Sum of the coefficients: {np.sum(x)}')      
            print(f'Residus:{np.round(error[0][0],6)}')
            print('')

        ########################

    
    if do_debug:
        x_star = x_list[-1]
        return x_star, s, lambda_, np.array(x_list), np.array(slack_list), np.array(lambda_list), rd_list, rb_list, rc_list, alpha_list, np.array(err_list)

    return x