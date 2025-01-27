import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.io import loadmat
from itertools import cycle
from scipy.sparse import csc_array
from Interior_Point import interior_point, interior_point2, interior_point3, interior_point4

import cProfile
import pstats

"""
File: code_ex_UNMIX.py
Author: Nils Foix-Colonier - Luca MIMOUNI - Baptiste LARDINOIT
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


def exemple1():
    ##############
    # Parameters #
    ##############
    N = 90 # number of spectra in the dictionary
    D, wv = load_A_and_wavelengths(N) # D has L (=113 wavelengths) rows, N columns (spectra)
    L = D.shape[0]
    K = 3 # sparsity --> number of nonzero coefficient i.e. activated spectra
    sigma = 0.0164 #1e-100 # noise amplitude, for instance 0.013 or 1e-100 (near 0, SNR about 2000 dB)

    #################
    # A Simple Case #
    #################
    do_simple_case = False
    if do_simple_case:
        N = 35
        D, wv = load_A_and_wavelengths(N)
        L = D.shape[0]
        K = 4
        sigma = 1e-100

    #################
    # Generate Data #
    #################

    ### Random seed set for reproducibility
    seed = 42
    np.random.seed(seed)

    ### Data generation
    x_gt = generate_x(K, N) # ground truth (K non-zero values choose between P spectras).
    y_gt = D@x_gt # noiseless signal
    y = y_gt + sigma*np.random.randn(L)
    y[y < 0] = 0. # even with strong noise, the sensor will never detect a negative amount of photons
    SNR = 10*np.log10(np.linalg.norm(y_gt)**2/(L*sigma**2))

    ###################
    # Find a Solution #
    ###################

    # With Least Square
    #x_star = np.linalg.inv(D.T @ D) @ D.T @ y # Least square solution

    # With Interior Point
        ##################
        # Initialisation #
        ##################
    A_ = np.eye(N)
    b_ = np.zeros((N,1))
    A_bar_ = np.ones((1,N))
    b_bar_ = np.array([1])
    G_ = D.T@D
    d_ = -(D.T@y).reshape(-1,1)
    x0_ = (1/N)*np.ones((N,1))
    iter_max = 100
    tol = 1e-5
        #######################
        # Choose Debug or Not #
        #######################
    # To test the algorithm performance, you should not use debug
    debug = False
    if debug:
        x_star, slack, lambda_, x_list, slack_list, lambda_list, rd_list, rb_list, rc_list, alpha_list, err_quadra_list = interior_point4(x0=x0_, G=G_, d=d_, A=A_, b=b_, A_bar=A_bar_, b_bar=b_bar_, y=y, D=D, tol=tol , iter_max=iter_max, do_debug=debug)
        plt.figure()
        plt.plot(np.squeeze(err_quadra_list), label='error')
        plt.title('error evolution')
        plt.xlabel('iter')
        plt.legend()
    else:
        x_star = interior_point3(x0=x0_, G=G_, d=d_, A=A_, b=b_, A_bar=A_bar_, b_bar=b_bar_, y=y, D=D, tol=tol , iter_max=iter_max, do_debug=debug)

    do_visualisation = True
    if do_visualisation:
        ##################
        # Compute errors #
        ##################
        err = 0.5 * np.linalg.norm(y.reshape(-1,1)-D@x_star, ord=2)**2
        err_gt = 0.5 * np.linalg.norm(y.reshape(-1,1)-D@x_gt, ord=2)**2
        print('err:\n', err) # value of the objective function at this point
        print('err_gt: \n', err_gt) # value of the objective function at this point


        #################
        # Visualisation #
        #################
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
    return x0_, G_, d_, A_, b_, A_bar_, b_bar_, y, D, tol, iter_max, debug



if __name__ == '__main__':
    x0_, G_, d_, A_, b_, A_bar_, b_bar_, y, D, tol, iter_max, debug = exemple1()
    cProfile.run("interior_point3(x0=x0_, G=G_, d=d_, A=A_, b=b_, A_bar=A_bar_, b_bar=b_bar_, y=y, D=D, tol=tol , iter_max=iter_max, do_debug=debug)", "profiling_results.prof")