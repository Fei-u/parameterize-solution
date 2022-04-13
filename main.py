#!/usr/bin/env python
# coding: utf-8

# In[1]:


from libs import *
from train import *
from PDEs import Heat
from plot import *


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# In[2]:


from NNs import FNN

layer_size = [3] + [100]*5 + [1]
net = FNN.net(layer_size=layer_size)

model = Heat.equation(net, 4, 1, 1)
model.to(device)

Train = Train(net, model, batch_size=2**8)
Train.train(epoch=10**5, lr=0.0001)

torch.save(net, 'net_model.pkl')

errors = Train.get_errors()

error_plt(errors)


# In[ ]:


from NNs import ResNN_F

net = ResNN_F.net(3, ResNN_F.basic_block, 100)
#收敛慢，震荡？？

model = Heat(net, 4, 1, 1)

Train = Train(net, model, batch_size=2**8)
Train.train(epoch=10**5, lr=0.0001)

torch.save(net, 'net_model.pkl')

errors = Train.get_errors()

error_plt(errors)

