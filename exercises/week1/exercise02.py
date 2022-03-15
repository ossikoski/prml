"""
Exercise Set 2: Estimation theory
1. LS and ML estimators
Team name in kaggle: Ossi Koski
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def main():
    # a)
    SSE_list_a, ML_list_a = find_f()
    sin, _, A, phi = sinusoid()
    #print(ML_list_a)
    #print((np.argmax(ML_list_a)/100)*0.5)  # TODO
    f_hat = (np.argmin(SSE_list_a)/100)*0.5  # TODO
    n = np.arange(160)
    sin_hat = A * np.cos(2 * np.pi * f_hat * n + phi)

    plot('a)', SSE_list_a, ML_list_a, sin, sin, sin_hat, f_hat)


    #b)
    """
    SSE_list_b, ML_list_b = find_f(samples=1000)
    plt.figure()
    plt.plot(SSE_list_b)
    plt.plot(ML_list_b)
    """


def sinusoid(noise = 0):
    """
    Form a sinusoidal signal
    Copied from assignment, edited

    Parameters:
    noise (int): noise to add to the sinusoid signal

    Returns:
    x0 (list[float]): sinusoid values without noise
    x (list[float]): sinusoid values with noise
    A (float): amplitude
    phi (float): phase
    """
    N = 160
    n = np.arange(N)
    f0 = 0.06752728319488948

    # Add noise to the signal
    sigmaSq = 0.0 + noise # 1.2
    phi = 0.6090665392794814
    A = 0.6669548209299414

    x0 = A * np.cos(2 * np.pi * f0 * n + phi)

    x = x0 + sigmaSq * np.random.randn(x0.size)

    return x0, x, A, phi


def find_f(error = 1.0, samples = 100):
    """
    Code that calculates SSE and ML to find f in range 0 -> 0.5

    Parameters:
    error (float): error for A_hat and phi_hat. 1.0 -> no error
    samples (int): Number of samples to test f for

    Returns:
    SSE_list (list[float]): All sums of squared errors
    ML_list (list[float]): All likelihood values
    """
    _, sin, A, phi = sinusoid()
    # Estimation parameters
    A_hat = A*error
    phi_hat = phi*error
    fRange = np.linspace(0, 0.5, samples)

    SSE_list = list()
    ML_list = list()
    for f in fRange:
        # compute SSE
        # compute likelihood
        # store them
        loss_SSE = 0
        p_total = 0
        for n, xn in enumerate(sin):
            sin_hat = A_hat * np.cos(2 * np.pi * f * n + phi)
            loss_SSE += np.square(xn - sin_hat)
            p_total *= stats.norm.pdf(f, sin_hat)
        SSE_list.append(loss_SSE)
        ML_list.append(p_total)

    return SSE_list, ML_list

def plot(title, SSE, ML, sin, sin_noise, sin_hat, f_hat, sigma2=0):
    """
    Plot four asked plots
    
    Parameters:
    title (string): Title of the whole subplot
    SSE (list): Sum of Squared Errors
    ML (list): Maximum Likelihood
    sin (list): original sinusoid
    sin_noise (list): original sinusoid with noise sigma2
    sin_hat (list): predicted sinusoid
    f_hat (float): predicted frequency
    sigma2 (float): Noise
    """
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(title)

    axs[0, 0].plot(SSE)
    axs[0, 0].set_title('Squared Error')

    axs[0, 1].plot(ML)
    axs[0, 1].set_title('Likelihood')

    axs[1, 0].plot(sin)
    axs[1, 0].plot(sin_noise, 'go')
    axs[1, 0].set_title(f'Signal and noisy samples (sigma2={sigma2})')

    axs[1, 1].plot(sin, 'b-')
    axs[1, 1].plot(sin_hat, 'r--')
    axs[1, 1].set_title(f'True f0=0.0675 (blue) and estimated f0={f_hat} (red)')

    plt.show()

"""
Extra:
How this really should be done?
Espexially if I need to estimate all A, phi, f0

Three nested for loops, all values of A, for every A 
check all possible phi etc
"""

if __name__ == '__main__':
    main()