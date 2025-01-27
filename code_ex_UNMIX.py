import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.io import loadmat
from itertools import cycle
from scipy.sparse import csc_array
from Interior_Point import interior_point2, interior_point3

"""
File: code_ex_UNMIX.py
Author: Nils Foix-Colonier
Date: 2025-01-06

Description: An introductive Python script to generate and visualize spectral unmixing problems for the UNMIX project.
"""


def load_A_and_wavelengths(P):
    """ Load the dictionnary of spectra with P (max 410) columns """
    Dic = loadmat('./spectra_USGS_ices_v2.mat') # todo change this path if needed
    A = Dic['speclib'][:, :P]
    wavelengths = Dic['wavelength'][:, 0] # axis for the values of the measurments
    return A, wavelengths


def generate_x(K, P, a_min=0.1):
    """ Return a vector x of size P with K nonzero positive random coefficients, also located randomly, with sum(x)=1 and nonzero values greater than a_min=0.1 """
    x_nz = -np.log(np.random.uniform(0, 1, K)) # when divided by its l1-norm this is a dirichlet distribution (ensuring an uniform distribution in the simplex volume)
    x_nz /= np.linalg.norm(x_nz, ord=1)
    x_nz = x_nz * (1 - K * a_min) + a_min # with that x min value is a_min and sum is 1 (1*(1-K*a_min) + K*a_min = 1)
    rd_idx = np.random.choice(P, K, replace=False)
    x = np.zeros(P)
    x[rd_idx] = x_nz
    return x

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

def exemple1():
    ### Parameters
    N = 110 # number of spectra in the dictionary
    D, wv = load_A_and_wavelengths(N) # D has L (=113 wavelengths) rows, N columns (spectra)
    L = D.shape[0]
    K = 3 # sparsity --> number of nonzero coefficient i.e. activated spectra
    sigma = 0.0164 #1e-100 # noise amplitude, for instance 0.013 or 1e-100 (near 0, SNR about 2000 dB)

    do_simple_case = False
    if do_simple_case:
        N = 35
        D, wv = load_A_and_wavelengths(N)
        L = D.shape[0]
        K = 4
        sigma = 1e-100

    ### Random seed set for reproducibility
    seed = 42
    np.random.seed(seed)

    ### Data generation
    x_gt = generate_x(K, N) # ground truth (K non-zero values choose between P spectras).
    y_gt = D@x_gt # noiseless signal
    y = y_gt + sigma*np.random.randn(L)
    y[y < 0] = 0. # even with strong noise, the sensor will never detect a negative amount of photons
    SNR = 10*np.log10(np.linalg.norm(y_gt)**2/(L*sigma**2))

    ### Computing a solution

    ##TODO: Interior Point
    #x_star = np.linalg.inv(D.T @ D) @ D.T @ y # Least square solution
    # A_ = np.block([[np.ones((2, N))],[np.eye(N)]])
    # b_ = np.block([[1],[-1],[np.zeros((N,1))]])
    A_ = np.eye(N)
    b_ = np.zeros((N,1))
    A_bar_ = np.ones((1,N))
    b_bar_ = np.array([1])
    G_ = D.T@D
    d_ = -(D.T@y).reshape(-1,1)
    x0_ = (1/N)*np.ones((N,1))
    # params = {"iter_max":50}
    debug = True
    iter_max = 100
    tol = 1e-5
    if debug:
        x_star, slack, lambda_, x_list, slack_list, lambda_list, rd_list, rb_list, rc_list, alpha_list, err_quadra_list = interior_point3(x0=x0_, G=G_, d=d_, A=A_, b=b_, A_bar=A_bar_, b_bar=b_bar_, y=y, D=D, tol=tol , iter_max=iter_max, do_debug=debug)
        plt.figure()
        plt.plot(np.squeeze(err_quadra_list), label='error')
        plt.title('error evolution')
        plt.xlabel('iter')
        plt.legend()
    else:
        x_star = interior_point3(x0=x0_, G=G_, d=d_, A=A_, b=b_, A_bar=A_bar_, b_bar=b_bar_, y=y, D=D, tol=tol , iter_max=iter_max, do_debug=debug)

    err = 0.5 * np.linalg.norm(y.reshape(-1,1)-D@x_star, ord=2)**2
    err_gt = 0.5 * np.linalg.norm(y.reshape(-1,1)-D@x_gt, ord=2)**2
    print('err:\n', err) # value of the objective function at this point
    print('err_gt: \n', err_gt) # value of the objective function at this point

    ### Plots
    plt.figure(figsize=(9, 9))
    # Plot the ground truth and the received signal
    plt.subplot(311)
    plt.plot(wv, y, 'b', linewidth=1, alpha=0.8, label=r'$y (Noised signal)$')
    plt.plot(wv, y_gt, 'g--', linewidth=1.2, label=r'$y_{gt} (Ground Truth)$')
    plt.plot(wv, D@x_star, 'r', label=r'$y_{pred} (prediction)$')
    plt.title("Received data, SNR = %d dB"%SNR); plt.ylabel("Amplitude"); plt.xlabel("Wavelength (µm)"); plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Plot the spectra involved
    plt.subplot(312)
    for p,x_p in enumerate(x_gt):
        if x_p >= 1e-15: # Treshold
            plt.plot(wv, D[:, p], label="Spectrum %d" % p)
    plt.title("Original atoms"); plt.ylabel("Amplitude"); plt.xlabel("Wavelength (µm)"); plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Plot the vector x_gt and the solution x_star
    plt.subplot(313)
    x_cut = x_gt; x_star_cut = x_star; x_cut[abs(x_cut) < 1e-15] = None; x_star_cut[abs(x_star_cut) < 1e-15] = None
    markerline, stemline, _ = plt.stem(x_cut, "g--", markerfmt="x", label="Truth"); plt.setp(stemline, linewidth=0.5); plt.setp(markerline, markersize=8)
    markerline, _, _ = plt.stem(x_star_cut, linefmt="r--", label="Solution found");  plt.setp(markerline, markersize=5)
    plt.title("Activated columns and their amplitudes, err = %.3e"%err); plt.ylabel("Coefficients values"); plt.xlabel("Index"); plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim([-0.5, N-0.5]); ax = plt.gca(); ax.xaxis.set_major_locator(MaxNLocator(integer=True)); plt.xticks(list(set(list(np.where(x_star>=1e-15)[0])+list(np.where(x_gt>1e-15)[0]) ))) # show xticks for included spectra only
    plt.tight_layout()
    plt.show()
    pass



if __name__ == '__main__':
    exemple1()