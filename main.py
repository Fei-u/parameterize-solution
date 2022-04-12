from libs import *
from train import *
from NNs import FNN
from PDEs import Heat
from plot import *

layer_size = [3] + [128]*4 + [1]
net = FNN.net(layer_size=layer_size)

model = Heat.equation(net, 4, 1, 1)

Train = Train(net, model, batch_size=2**8)
Train.train(epoch=10**5, lr=0.0001)

torch.save(net, 'net_model.pkl')

errors = Train.get_errors()
