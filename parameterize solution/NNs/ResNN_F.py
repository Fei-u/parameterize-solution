from turtle import forward
from numpy import block
from torch import sigmoid
from libs import *

class net(nn.Module):
    def __init__(self, input_dim, block, layer_dim):
        super(net, self).__init__()
    
        self.layer_in = nn.Linear(input_dim, layer_dim)
        
        self.block1 = block(layer_dim, layer_dim)
        self.block2 = block(layer_dim, layer_dim)
        
        self.layer_out = nn.Linear(layer_dim, 1)
        
    def forward(self, x):
        out = F.sigmoid(self.layer_in(x)) # warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
        out = self.block1(out)
        out = self.block2(out)
        out = self.layer_out(out)

        return out
    
    
class basic_block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(basic_block, self).__init__()

        self.layer_1 = nn.Linear(input_dim, output_dim)
        self.layer_2 = nn.Linear(output_dim, output_dim)
        self.layer_3 = nn.Linear(output_dim, output_dim)
        
    def forward(self, x):
        out = F.sigmoid(self.layer_1(x))
        out = F.sigmoid(self.layer_2(out))
        out = F.sigmoid(self.layer_3(out))
        
        out = self.layer_1(x) + out
        
        return out


        
        