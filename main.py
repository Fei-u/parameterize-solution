from libs import *
from train import *
from PDEs import Heat
from plot import *

from NNs import FNN
from NNs import ResNN_F

layer_size = [3] + [100]*5 + [1]
net = FNN.net(layer_size=layer_size)
net.to(device)

model = Heat.equation(net, 4, 1, 1)

Train_1 = Train(net, model, batch_size=2**8)
Train_1.train(epoch=10**5, lr=0.0001)
torch.save(net, 'net_model.pkl')

errors = Train_1.get_errors()

error_plt(errors)


net = ResNN_F.net(3, ResNN_F.basic_block, 100)
#收敛慢，震荡？？

Train_2 = Train(net, model, batch_size=2**8)
Train_2.train(epoch=10**5, lr=0.0001)

torch.save(net, 'net_model.pkl')

errors = Train_2.get_errors()

error_plt(errors)

