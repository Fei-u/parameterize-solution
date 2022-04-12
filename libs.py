import imp
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import collections

"""conda support"""

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    