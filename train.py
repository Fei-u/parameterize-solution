from numpy import minimum
from libs import *

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
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0.00001)
        avg_loss = 0
        
        for i in range(epoch):
            optimizer.zero_grad()
            loss = self.model.loss_func(self.batch_size)
            avg_loss = avg_loss + float(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if i % 500 == 0 and i != 0:
                loss = avg_loss/500
                print("Epoch {} - lr {} -  loss: {}".format(int(i/500), lr, loss))
                avg_loss = 0

                error = self.model.loss_func(2**8)
                self.errors.append(error.detach())

    def get_errors(self):
        return self.errors
    