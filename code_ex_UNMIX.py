import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.io import loadmat
from itertools import cycle

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

def interior_point(x0, G, d, A, b, y, D, iter_max):
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

    while iter < iter_max:
        # Calcul des résidus
        mu = (1 / m) * ((s.T).dot(lambda_)) # duality measure
        rd = G.dot(x) - A.T.dot(lambda_) + d # stationarité
        rb = A.dot(x) - s - b
        rc = lambda_ * s - sigma * mu 
        residus = np.block([[rd],
                            [rb],
                            [rc]])
    
        Jacobienne = np.block([[G, -A.T , np.zeros((n, m))],
                               [A, np.zeros((m, m)), -np.eye(m)],
                               [np.zeros((m,n)), np.diag(s[:, 0]), np.diag(lambda_[:, 0])]]) #y[:, 0]
        
        delta = np.linalg.solve(Jacobienne, residus)

        delta_x = delta[:n]
        delta_lambda_ = delta[n:n+m]
        delta_s = delta[n+m:]

        # Compute alpha to ensure feasibility, all(y) > 0 and all(lambda_) > 0
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
            alpha = alpha_coef * alpha_max

        # print(f"{y=}")


        # Mise a jours des variables
        x       = x       - alpha * delta_x
        s       = s       - alpha * delta_s
        lambda_ = lambda_ - alpha * delta_lambda_

        # print(f"{alpha=}")
        # print(f"{A.dot(x) - b=}")
        # print(f"{y=}")

        # Debugging
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


        if iter % 10:
            print(iter)
            #print(f"Jacobienne.shape : {Jacobienne.shape}; residus.shape : {residus.shape}")
            #print(f"rd.shape : {rd.shape}, rb.shape : {rb.shape}, rc.shape : {rc.shape}")
            #print(f"delta_x.shape : {delta_x.shape}, delta_y.shape : {delta_y.shape}, delta_lambda_.shape : {delta_lambda_.shape}")
        x_star = x_list[-1]

        iter += 1
    
    return x_star, s, lambda_, np.array(x_list), np.array(slack_list), np.array(lambda_list), rd_list, rb_list, rc_list, alpha_list, np.array(err_quadra_list), np.array(err_norm_list)

def exemple1():
    ### Parameters
    P = 111 # number of spectra in the dictionary
    D, wv = load_A_and_wavelengths(P) # A has N (=113 wavelengths) rows, P columns (spectra)
    N = D.shape[0]
    K = 4 # sparsity --> number of nonzero coefficient i.e. activated spectra
    sigma = 0.013 #1e-100 # noise amplitude, for instance 0.013 or 1e-100 (near 0, SNR about 2000 dB)

    do_simple_case = False
    if do_simple_case:
        P = 30
        D, wv = load_A_and_wavelengths(P)
        N = D.shape[0]
        K = 4
        sigma = 1e-100

    ### Random seed set for reproducibility
    seed = 42
    np.random.seed(seed)

    ### Data generation
    x_gt = generate_x(K, P) # ground truth (K non-zero values choose between P spectras).
    y_gt = D@x_gt # noiseless signal
    y = y_gt + sigma*np.random.randn(N)
    y[y < 0] = 0. # even with strong noise, the sensor will never detect a negative amount of photons
    SNR = 10*np.log10(np.linalg.norm(y_gt)**2/(N*sigma**2))

    ### Computing a solution

    ##TODO: Interior Point
    #x_star = np.linalg.inv(D.T @ D) @ D.T @ y # Least square solution
    A_ = np.block([[np.ones((2, P))],[np.eye(P)]])
    b_ = np.block([[1],[-1],[np.zeros((P,1))]])
    x0_ = (1/P)*np.ones((P,1))
    # params = {"iter_max":50}
    x_star, slack, lambda_, x_list, slack_list, lambda_list, rd_list, rb_list, rc_list, alpha_list, err_quadra_list, err_norm_list = interior_point(x0=x0_, G=(D.T).dot(D), d=(-(D.T).dot(y)).reshape(-1,1), A=A_, b=b_, y=y, D=D, iter_max = 50)
    
    #x_star[0] = 0.5
    err = 0.5 * np.linalg.norm(y.reshape(-1,1)-D@x_star, ord=2)**2
    err_gt = 0.5 * np.linalg.norm(y.reshape(-1,1)-D@x_gt, ord=2)**2
    print('err ', err) # value of the objective function at this point
    print('err_gt', err_gt) # value of the objective function at this point

    plt.figure()
    plt.plot(np.squeeze(err_quadra_list), label='w/ quadra')
    plt.plot(np.squeeze(err_norm_list), label='w/ norm')
    plt.title('err')
    plt.legend()

    # ## Visualise function
    # do_visu = False
 

    # if do_visu:

    #     # Parameters
    #     n=20
    #     p=0.4

    #     colors = plt.cm.jet(np.linspace(0,1,115))# Initialize holder for trajectories
    #     xtraj=np.zeros(n+1,float)
        
    #     plt.figure()
    #     for i, slack in enumerate(slack_list):
    #         plt.plot(slack.squeeze(), color=colors[i])
        
    #     plt.title('Slack evolution')
    #     plt.show()

    #     stop
    #     plt.plot(np.array(lambda_list[:,0]).squeeze(), label='lambda',  color='g')
    #     plt.plot(alpha_list[0],  label='alpha',  color='b')
    #     plt.title("Evolution of y and lambda")
    #     plt.legend()

    #     plt.subplot(312)
    #     plt.plot(rd_list)
    #     plt.plot(rb_list)
    #     plt.plot(rc_list)
    #     plt.title('Evolution of the norm of residus'    )
    #     plt.legend()

    #     plt.subplot(313)
    #     plt.plot(np.arange(len(err_list)), err_list)
    #     plt.title('err')

    #     plt.tight_layout()
    #     plt.show()

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
    plt.xlim([-0.5, P-0.5]); ax = plt.gca(); ax.xaxis.set_major_locator(MaxNLocator(integer=True)); plt.xticks(list(set(list(np.where(x_star>=1e-15)[0])+list(np.where(x_gt>1e-15)[0]) ))) # show xticks for included spectra only
    plt.tight_layout()
    plt.show()
    pass



if __name__ == '__main__':
    exemple1()