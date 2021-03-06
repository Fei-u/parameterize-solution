import matplotlib.pyplot as plt
import numpy as np
import torch

def error_plt(errors, i):
    plt.figure(i)
    plt.plot(np.log10(errors), "-b", label="10^(x)")
    plt.savefig("error_{}.png".format(i))
    plt.show()