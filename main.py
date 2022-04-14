from libs import *
from train import *
from PDEs import Heat
from plot import *

from NNs import FNN
from NNs import ResNN_F

layer_size = [3] + [100]*5 + [1]
net_1 = FNN.net(layer_size=layer_size)
net_1.to(device)

model = Heat.equation(net_1, 4, 1, 1)

Train_1 = Train(net_1, model, batch_size=2**8)
Train_1.train(epoch=10**5, lr=0.0001)
torch.save(net_1, 'net_model.pkl')

errors_1 = Train_1.get_errors()

error_plt(errors_1, 1)


net_2 = ResNN_F.net(3, ResNN_F.basic_block, 100)
#收敛慢，震荡？？
net_2.to(device)

model = Heat.equation(net_2, 4, 1, 1)

Train_2 = Train(net_2, model, batch_size=2**8)
Train_2.train(epoch=10**5, lr=0.0001)

torch.save(net_2, 'net_model.pk2')

errors_2 = Train_2.get_errors()

error_plt(errors_2, 2)

