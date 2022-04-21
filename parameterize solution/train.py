from numpy import minimum
from all_libs import *
from tqdm import tqdm
from time import sleep

"""_summary_
    the train process of NN
    
   _inputs_:
        _type_: _description_
        net : network structure
        model : PDE function model
        batch_size : ...
        epoch : ...
        lr : learning rate
"""

class Train():
    def __init__(self, net, model, batch_size):
        self.errors = []
        self.batch_size = batch_size
        self.net = net
        self.model = model

    def train(self, epoch, lr):
        optimizer = optim.Adam(self.net.parameters(), lr)
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0.00001)
        t = tqdm(range(epoch))
        
        for i in t:
            optimizer.zero_grad()
            loss = self.model.loss_func(self.batch_size)
            loss.backward()
            optimizer.step()
         #   scheduler.step()
         
            t.set_postfix(loss=format(loss,'.3f'))
            
            error = self.model.loss_func(2**8)
            self.errors.append(error.detach())

    def get_errors(self):
        return self.errors
    