"""
Exercise Set 3: Detection theory
1. Implement a sinusoid detector
Ossi Koski
"""

from charset_normalizer import detect
import matplotlib.pyplot as plt
import numpy as np


def main():
    sin, exp, y, y_n = create_signal()

    # First detector
    det1 = np.convolve(y_n, np.flip(sin), 'same')
    plot(y, y_n, det1)
    
    # Second detector
    det2 = np.convolve(y_n, np.flip(exp), 'same')
    plot(y, y_n, np.abs(det2))

def create_signal():
    """
    a) & b): Create asked signals

    Returns
    y (): Noiseless signal
    y_n (): Noisy signal
    """
    f = 0.1
    n = np.arange(500, 600)
    sin = np.cos(2 * np.pi * f * n)

    y = np.concatenate((np.zeros(500), sin, np.zeros(300)))
    y_n = y + np.sqrt(0.5) * np.random.randn(y.size)

    exp = np.exp(-2 * np.pi * 1j * f * n)

    return sin, exp, y, y_n

def plot(y, y_n, d):
    """
    plot i) the noise free signal x[n], 
    ii) noisy x[n] + w[n] and 
    iii) detector output
    """
    fig, axs = plt.subplots(3, 1)

    axs[0].plot(y)
    axs[0].set_title('Noiseless Signal')

    axs[1].plot(y_n)
    axs[1].set_title('Noisy Signal')

    axs[2].plot(d)
    axs[2].set_title('Detection Result')

    plt.show()

if __name__ == '__main__':
    main()
