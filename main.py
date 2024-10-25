import numpy as np
import torch
import argparse

from train import DeepBSDEOSrHeston
import time
from scipy.stats import norm, sem

import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description="parameters")
    parser.add_argument('--num_time_step', default=100, type=int, help="number of time steps")
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--lr', default=1, type=float, help="base learning rate")
    parser.add_argument('--num_iterations', default=200, type=int, help="number of iterations in each epoch")
    parser.add_argument('--num_epochs', default=1, type=int, help="epoch for training")
    parser.add_argument('--valid_size', default=2 ** 20, type=int, help="number of samples for lower bound")
    parser.add_argument('--upper_size', default=2 ** 15, type=int, help="number of samples for upper bound")
    parser.add_argument('--logging_frequency', default=20, type=int, help="frequency of displaying results")
    parser.add_argument('--device', default='cpu', type=str, help="cpu")
    # rough heston model
    parser.add_argument('--option_name', default='RoughHeston', type=str, help="types of option")
    parser.add_argument('--lambda', default=0.3, type=float)
    parser.add_argument('--nu', default=0.3, type=float)
    parser.add_argument('--theta', default=0.02, type=float)
    parser.add_argument('--v0', default=0.02, type=float)
    parser.add_argument('--expiration', default=1.0, type=float, help="time horizon")
    parser.add_argument('--rho', default=-0.7, type=float)
    parser.add_argument('--s_init', default=100, type=float)
    parser.add_argument('--r', default=0.06, type=float)
    parser.add_argument('--d', default=2, type=int, help="dimension of Brownian motions")
    parser.add_argument('--strike', default=105.0, type=float, help="strike price of the option")
    args = parser.parse_args()
    args = {**vars(args)}
    print(''.join(['='] * 80))
    tplt = "{:^20}\t{:^20}\t{:^20}"
    print(tplt.format("Name", "Value", "Type"))
    for k, v in args.items():
        print(tplt.format(k, v, str(type(v))))
    print(''.join(['='] * 80))
    return args


class SDE(object):
    def __init__(self, dim, expiration, num_time_step):
        self.dim = dim
        self.time_horizon = expiration
        self.num_time_step = num_time_step
        self.dt = self.time_horizon / self.num_time_step
        self.sqrt_dt = torch.sqrt(torch.tensor(self.dt))


class RoughHeston(SDE):
    def __init__(self, dim, expiration, num_time_step, cfg):
        super(RoughHeston, self).__init__(dim, expiration, num_time_step)
        self.lam = cfg['lambda']
        self.nu = cfg['nu']
        self.theta = cfg['theta']
        self.v0 = cfg['v0']
        self.nodes = nodes
        self.weights = weights
        self.rho = cfg['rho']
        self.s_init = cfg['s_init']
        self.rate = cfg['r']
        self.strike = cfg['strike']
        self.num_factor = self.nodes.shape[0]
        self.A = torch.eye(self.num_factor) + torch.diag(self.nodes) * self.dt + self.lam * self.weights.unsqueeze(1) * self.dt

    def sample(self, num_sample):
        dw_sample = self.sqrt_dt * torch.randn([num_sample, self.dim, self.num_time_step], dtype=torch.float32)
        s_sample = torch.zeros([num_sample, self.num_time_step + 1], dtype=torch.float32)
        v_sample = torch.zeros([num_sample, self.num_time_step + 1], dtype=torch.float32)
        s_sample[:, 0] = self.s_init

        v_factors = torch.zeros([num_sample, self.num_factor, self.num_time_step + 1], dtype=torch.float32)
        v_init = self.v0 / self.nodes / torch.sum(self.weights / self.nodes)

        v_now = torch.tile(v_init, (num_sample, 1))
        x_now = np.log(self.s_init) * torch.ones(num_sample)
        v_factors[:, :, 0] = v_now
        v_sample[:, 0] = torch.dot(v_init, self.weights)

        A_inv = np.linalg.inv(self.A)
        b = self.theta * self.dt + (self.nodes * v_init) * self.dt
        sq_rho = np.sqrt(1.0 - self.rho ** 2)
        for i in range(self.num_time_step):
            sq_v_now = torch.maximum(v_now @ self.weights.unsqueeze(1), torch.tensor([0.0])).sqrt().squeeze()
            x_now += self.rate * self.dt + sq_v_now * (self.rho * dw_sample[:, 1, i] + sq_rho * dw_sample[:, 0, i]) \
                    - 0.5 * sq_v_now ** 2 * self.dt
            v_now = (v_now + self.nu * (sq_v_now * dw_sample[:, 1, i]).unsqueeze(1) + b) @ A_inv

            s_sample[:, i+1] = torch.exp(x_now)
            v_factors[:, :, i+1] = v_now
            v_sample[:, i+1] = torch.maximum(v_now @ self.weights.unsqueeze(1), torch.tensor([0.0])).squeeze()

        return dw_sample, s_sample, v_sample

    def payoff(self, s):
        ans = torch.maximum(self.strike - s, torch.tensor([0.0]))
        discount = torch.ones(s.shape)
        discount[:, 0] = torch.zeros((s.shape[0]))
        discount = torch.exp(-self.rate * self.dt * torch.cumsum(discount, dim=1))
        return discount * ans

    def phi(self, s):
        """
        :return: phi(s), where payoff = max(phi(s), 0)
        """
        ans = self.strike - s
        discount = torch.ones(s.shape)
        discount[:, 0] = torch.zeros((s.shape[0]))
        discount = torch.exp(-self.rate * self.dt * torch.cumsum(discount, dim=1))
        return discount * ans

    def phi_partial(self, s, delta_t):
        """
        compute phi(s) using paths on partial time points with time difference delta_t.
        :return: phi(s), where payoff = max(phi(s), 0)
        """
        ans = self.strike - s
        discount = torch.ones(s.shape)
        discount[:, 0] = torch.zeros((s.shape[0]))
        discount = torch.exp(-self.rate * delta_t * torch.cumsum(discount, dim=1))
        return discount * ans


def get_sde(cfg):
    try:
        return globals()[cfg['option_name']](cfg['d'], cfg['expiration'], cfg['num_time_step'], cfg)
    except KeyError:
        raise KeyError("Option type required not found. Please try others.")


def true_lower(cfg, c_fun, lower_size=2 ** 20):
    """
    compute true lower bound
    """
    option = get_sde(cfg)
    num_time_step = cfg['num_time_step']

    test_start_time = time.time()
    _, x_valid, v_valid = option.sample(lower_size)
    phi_valid = option.phi(x_valid)
    x_valid = x_valid.to(cfg['device'])
    phi_valid = phi_valid.to(cfg['device'])

    payout_valid = torch.maximum(phi_valid[:, num_time_step], torch.tensor([0.0]))
    for k in range(num_time_step - 1, 0, -1):
        with torch.no_grad():
            func = c_fun[k].eval()
            xx_valid_k = torch.cat((phi_valid[:, k].unsqueeze(1), x_valid[:, k].unsqueeze(1), v_valid[:, k].unsqueeze(1)), dim=1)
            continue_valid = func(xx_valid_k).reshape((lower_size,))
            exercise_valid = torch.maximum(phi_valid[:, k], torch.tensor([0.0]))
            idx_valid = (exercise_valid >= continue_valid) & (exercise_valid > 0)
            payout_valid[idx_valid] = exercise_valid[idx_valid]
    price_lower_bound, se = torch.mean(payout_valid), sem(payout_valid)

    confidence = 0.95
    z = norm.ppf((1 + confidence) / 2)
    print(f"-------lower bound evaluation time is {time.time() - test_start_time:.2f}")
    return price_lower_bound, se * z


def true_upper_rough_heston(cfg, g_fun, c1_fun, weights, scale=32, upper_size=2 ** 15):
    """
    true upper bound via non-nested Monte Carlo
    """
    delta_t = cfg['expiration'] / cfg['num_time_step']                              # original \Delta t
    num_fine_grid = scale * cfg['num_time_step']
    cfg['num_time_step'] = num_fine_grid
    rho = cfg['rho']
    sq_rho = torch.sqrt(torch.tensor(1 - rho ** 2))
    nu = cfg['nu']
    vol_mat_T = torch.tensor([[rho, nu * torch.sum(weights)], [sq_rho, 0.0]])
    option_upper = get_sde(cfg)

    upper_start_time = time.time()
    dw_upper, x_upper, v_upper = option_upper.sample(upper_size)   # generate samples on fine time grid
    # only need payoff on coarse grid
    phi_required = option_upper.phi_partial(x_upper[:, 0:(num_fine_grid + 1):scale], delta_t)
    dm_mat = torch.zeros(size=(upper_size, num_fine_grid), dtype=torch.float32, device=cfg['device'])

    for j in range(scale, num_fine_grid):
        k = np.floor(j / scale)
        z = torch.sqrt(v_upper[:, j]).reshape((upper_size, 1)) * (dw_upper[:, :, j] @ vol_mat_T)  # \sigma(S_k) dW_k
        with torch.no_grad():
            grad_model = g_fun[k].eval()
            xv_upper_j = torch.cat((x_upper[:, j].unsqueeze(1), v_upper[:, j].unsqueeze(1)), dim=1)
            g = grad_model(xv_upper_j)
        dm_mat[:, j] = torch.sum(torch.mul(g, z), dim=1)

    with torch.no_grad():
        c1_model = c1_fun.eval()
        xx_valid_1 = torch.cat((phi_required[:, 1].unsqueeze(1), x_upper[:, scale].unsqueeze(1), v_upper[:, scale].unsqueeze(1)), dim=1)
        v1 = torch.maximum(c1_model(xx_valid_1).reshape((upper_size,)), phi_required[:, 1])  # should be positive
        dm_mat[:, scale - 1] = v1 - torch.mean(v1)
    m_total = torch.cumsum(dm_mat, dim=1)
    m_total = torch.cat([torch.zeros(size=(upper_size, 1), dtype=torch.float32).to(cfg['device']), m_total], dim=1)
    martingale = m_total[:, 0:(num_fine_grid + 1):scale]

    v_mat = torch.maximum(phi_required, torch.tensor([0.0])) - martingale
    m_star, k_star = torch.max(v_mat, dim=1)
    upper_data = v_mat[range(upper_size), k_star]
    price_upper_bound, se = torch.mean(upper_data), sem(upper_data)

    confidence = 0.95
    z = norm.ppf((1 + confidence) / 2)
    print(f"-------upper bound evaluation time is {time.time() - upper_start_time:.2f}")
    return price_upper_bound, se * z


if __name__ == '__main__':
    # setup
    nodes = torch.tensor([0.05, 8.71708699])
    weights = torch.tensor([0.76732702, 3.22943184])
    cfg = get_args()
    option = get_sde(cfg)

    # Bermudan option price
    pricer = DeepBSDEOSrHeston(cfg, option, weights)
    c_fun, g_fun, V0 = pricer.train()
    # print(V0)

    print("\n\n")
    print("-----------------results for Bermudan option-------------------------")
    option_price, h_lower = true_lower(cfg, c_fun, lower_size=cfg['valid_size'])
    print(f"lower bound is {option_price:.4f}, confidence interval: +-{h_lower:.4f}")

    c1_fun = c_fun[1]
    option_upper, h_upper = true_upper_rough_heston(cfg, g_fun, c1_fun, weights, scale=32, upper_size=cfg['upper_size'])
    print(f"upper bound is {option_upper:.4f}, confidence interval: +-{h_upper:.4f}")

    print(f"95% confidence interval is [{option_price - h_lower:.4f}, {option_upper + h_upper:.4f}].")

    print("\n")
    print("---------------ref: results for European option----------------------")
    # European option price via Monte Carlo
    M = 10000
    dw, s, v = option.sample(M)

    r = cfg['r']
    strike = cfg['strike']
    T = cfg['expiration']
    euro_op_price = np.exp(-r * T) * torch.mean(torch.maximum(strike - s[:, -1], torch.tensor([0.0])))
    print(f"The price of European option is {euro_op_price:.4f}.")

    # # plot some paths
    # times = np.arange(cfg["num_time_step"] + 1)
    # plt.figure()
    # plt.plot(times, v[0:30, :].T)
    # plt.show()

