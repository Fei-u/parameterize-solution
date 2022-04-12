from turtle import forward
from torch import sigmoid
from libs import *

class F_net(nn.Module):
    """_summary_
        this is a Resnet (full_connected version)
    Args:
        ResNN (_type_): _description_
        ayer_size : the dimension of each layer
    """
    def __init__(self, layer_size):
        super(F_net, self).__init__()
        
        self.layer_size = layer_size
        self.layer = nn.Sequential()
        
    def out(self, layer, x):
        out = layer(x)
        return out
    
    def blcok_basic(self, a, b, c):
        block_basic = nn.Sequential(
            nn.Linear(a, b),
            nn.Sigmoid(),
            nn.Linear(b, c),
        )
        return block_basic
        
    def forward(self, x):
        layer_input = nn.Sequential(
            nn.Linear(self.layer_size[0], self.layer_size[1]),
            nn.Sigmoid()
        )
        out = self.out(layer_input, x)
        
        for i in range(1, (len(self.layer_size)-2)/ 3):
            out = self.out( 
                        self.blcok_basic(self.layer_size[i], self.layer_size[i+1], self.layer_size[i+2]),
                        out
                    )
            layer_i = nn.Sequential(
                nn.Linear(self.layer_size[i+2], self.layer_size[i+3]),
                nn.Sigmoid()
            ) 
            out = self.out(layer_i, out) + out
            layer_middle = nn.Sequential(layer_middle, layer_i)
        
        layer_output = nn.Sequential(
            nn.Linear(self.layer_size[-2], self.layer_size[-1])
        )
        out = self.out(layer_output, out)
        self.layer = nn.Sequential(layer_input, layer_middle, layer_output) 

        return out