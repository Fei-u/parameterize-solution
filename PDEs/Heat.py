from libs import *

import torch.nn

class equation():
    def __init__(self, net, te, xe, ye):
        self.net = net
        self.te = te
        self.xe = xe
        self.ye = ye

    def sample(self, size=2**8):
        te = self.te
        xe = self.xe
        ye = self.ye
        
        x = torch.cat((torch.rand([size, 1]) * te, torch.rand([size, 1]) * xe, torch.rand([size, 1]) * ye), dim=1).to(device)
        x_initial = torch.cat((torch.zeros(size, 1), torch.rand([size, 1]) * xe, torch.rand([size, 1]) * ye), dim=1).to(device)
        x_boundary_1 = torch.cat((torch.rand([size, 1]) * te, torch.zeros([size, 1]), torch.rand(size, 1) * ye), dim=1).to(device)
        x_boundary_2 = torch.cat((torch.rand([size, 1]) * te, torch.ones([size, 1]) * xe, torch.rand(size, 1) * ye), dim=1).to(device)
        x_boundary_3 = torch.cat((torch.rand([size, 1]) * te, torch.rand([size, 1]) * xe, torch.ones(size, 1) * ye), dim=1).to(device)
        x_boundary_4 = torch.cat((torch.rand([size, 1]) * te, torch.rand([size, 1]) * xe, torch.zeros(size, 1)), dim=1).to(device)
        return x, x_initial, x_boundary_1, x_boundary_2, x_boundary_3, x_boundary_4

    def loss_func(self, size=2**8):

        x, x_initial, x_boundary_1, x_boundary_2, x_boundary_3, x_boundary_4 = self.sample(size=size)
        x = Variable(x, requires_grad=True)
        
        d = torch.autograd.grad(self.net(x), x, grad_outputs=torch.ones_like(self.net(x)), create_graph=True,)
        dt = d[0][:, 0].reshape(-1, 1)  # transform the vector into a column vector
        dx = d[0][:, 1].reshape(-1, 1)
        dy = d[0][:, 2].reshape(-1, 1)
        # du/dxdx
        dxx = torch.autograd.grad(dx, x, grad_outputs=torch.ones_like(dx), create_graph=True)[0][:, 1].reshape(-1, 1)
        # du/dydy
        dyy = torch.autograd.grad(dy, x, grad_outputs=torch.ones_like(dy), create_graph=True)[0][:, 2].reshape(-1, 1)

        f = np.pi * (torch.cos(np.pi*x[:, 0])) * (torch.sin(np.pi*x[:, 1])) * (torch.sin(np.pi*x[:, 2]))\
             + 2 * np.pi * np.pi * (torch.sin(np.pi*x[:, 0])) * (torch.sin(np.pi*x[:, 1])) * (torch.sin(np.pi*x[:, 2]))

        diff_error = (dt - dxx - dyy - f.reshape(-1, 1))**2
        # initial condition
        init_error = (self.net(x_initial)) ** 2
        # boundary condition
        bd_left_error = (self.net(x_boundary_1)) ** 2
        bd_right_error = (self.net(x_boundary_2)) ** 2
        bd_up_error = (self.net(x_boundary_3)) ** 2
        bd_down_error = (self.net(x_boundary_4)) ** 2

        return torch.mean(diff_error + init_error + bd_left_error + bd_right_error + bd_up_error + bd_down_error).to(device="cpu")