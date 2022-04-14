import matplotlib.pyplot as plt
import numpy as np
import torch

def error_plt(errors):
    plt.plot(np.log10(errors), "-b", label="10^(x)")
    plt.show()