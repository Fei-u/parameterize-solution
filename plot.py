import matplotlib.pyplot as plt
import numpy as np

def error_plt(error):
    plt.plot(np.log10(error), "-b", label="10^(x)")
    plt.show()