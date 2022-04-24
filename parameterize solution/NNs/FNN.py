from turtle import forward
from numpy import blackman, block

from torch import batch_norm
from libs import *

class net(nn.Module):
    """_summary_
        this is a full_connected neural network
    Args:
        FNN (_type_): _description_
            layer_size : the dimension of each layer
    """
    def __init__(self, layer_size):
        super(net, self).__init__()
        
        self.layer_list = []
        for i in range(len(layer_size)-2):
            self.layer_list.append(("layer_%d" %i, nn.Linear(layer_size[i], layer_size[i+1])))
            self.layer_list.append(("activate_%d" %i, nn.Sigmoid()))
        
        self.layer_list.append(("layer_output", nn.Linear(layer_size[-2], layer_size[-1])))
        self.layer_dir = collections.OrderedDict(self.layer_list)
        self.layer = nn.Sequential(self.layer_dir)
        
    def forward(self, x):
        return self.layer(x)


        
        