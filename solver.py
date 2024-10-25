import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_out)
        self.bn0 = nn.BatchNorm1d(n_in)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.bn0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class BSDEModelrHeston(nn.Module):
    def __init__(self, cfg, option, k, weights, init_c=None, init_grad=None):
        super(BSDEModelrHeston, self).__init__()
        self.option = option
        self.dt = option.dt
        self.hidden_dim = cfg['hidden_dim']
        self.rho = cfg['rho']
        self.sq_rho = torch.sqrt(torch.tensor(1 - self.rho ** 2))
        self.nu = cfg['nu']
        self.weights = weights
        self.vol_mat_T = torch.tensor([[self.rho, self.nu * torch.sum(self.weights)], [self.sq_rho, 0.0]])
        self.num_time_step = cfg['num_time_step']
        self.d = cfg['d']  # d == 2
        self.k = k
        self.c_network = MLP(self.d + 1, 1, self.hidden_dim)  # input is [phi, x, v]
        self.grad_network = MLP(self.d, self.d, self.hidden_dim)
        if init_c is not None:
            self.c_network.load_state_dict(init_c.state_dict(), strict=False)
        else:
            nn.init.xavier_uniform_(self.c_network.fc1.weight)
            nn.init.xavier_uniform_(self.c_network.fc2.weight)
            nn.init.xavier_uniform_(self.c_network.fc3.weight)
        if init_grad is not None:
            self.grad_network.load_state_dict(init_grad.state_dict(), strict=False)
        else:
            nn.init.xavier_uniform_(self.grad_network.fc1.weight)
            nn.init.xavier_uniform_(self.grad_network.fc2.weight)
            nn.init.xavier_uniform_(self.grad_network.fc3.weight)

    def forward(self, xx, dw, y_in, tau):   # xx = (phi, x, v), phi=(K-x) represents the additional input feature.
        input_size = xx.shape[0]
        z = torch.sqrt(xx[:, 2]).reshape((input_size, 1)) * (dw[:, :] @ self.vol_mat_T)  # \sigma(X_k) dW_k
        g = self.grad_network(xx[:, 1:])
        y_k = torch.sum(torch.mul(g, z), dim=1, keepdim=True)

        if self.k == self.num_time_step - 1:
            y_out = y_k
            cv_out = self.c_network(xx)
            output = cv_out + y_out
        else:
            y_out = torch.cat([y_k, y_in], dim=1)
            ans = torch.cumsum(y_out, dim=1)
            y_sum = ans[range(input_size), tau - 1 - self.k].reshape((input_size, 1))
            cv_out = self.c_network(xx)
            output = cv_out + y_sum
        return output, cv_out, y_out


if __name__ == '__main__':

    c_net = MLP(3, 1, 32)
    xx = torch.rand([10, 3])
    out = c_net(xx)
    print(out)

