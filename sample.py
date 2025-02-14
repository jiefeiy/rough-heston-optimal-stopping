import numpy as np
import argparse



def get_args():
    parser = argparse.ArgumentParser(description="parameters")    
    # rough heston model
    parser.add_argument('--num_time_step', default=100, type=int, help="number of time steps")
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



class RoughHeston(object):
    def __init__(self, nodes, weights,cfg):
        super(RoughHeston, self).__init__()
        self.lam = cfg['lambda']
        self.nu = cfg['nu']
        self.theta = cfg['theta']
        self.v0 = cfg['v0']
        self.nodes = nodes
        self.weights = weights
        self.dim = cfg['d']
        self.time_horizon = cfg["expiration"]
        self.num_time_step = cfg["num_time_step"]
        self.dt = self.time_horizon / self.num_time_step
        self.sqrt_dt = np.sqrt(np.array(self.dt))

        self.rho = cfg['rho']
        self.s_init = cfg['s_init']
        self.rate = cfg['r']
        self.strike = cfg['strike']
        self.num_factor = self.nodes.shape[0]
        self.A = np.eye(self.num_factor) + np.diag(self.nodes) * self.dt + self.lam * self.weights[:, np.newaxis] * self.dt

    def sample(self, num_sample):
        dw_sample = self.sqrt_dt * np.random.normal(size = (num_sample, self.dim, self.num_time_step))
        s_sample = np.zeros((num_sample, self.num_time_step + 1))
        v_sample = np.zeros((num_sample, self.num_time_step + 1))
        s_sample[:, 0] = self.s_init

        v_factors = np.zeros((num_sample, self.num_factor, self.num_time_step + 1))
        v_init = self.v0 / self.nodes / np.sum(self.weights / self.nodes)

        v_now = np.tile(v_init, (num_sample, 1))
        x_now = np.log(self.s_init) * np.ones(num_sample)
        v_factors[:, :, 0] = v_now
        v_sample[:, 0] = np.dot(v_init, self.weights)

        A_inv = np.linalg.inv(self.A)
        b = self.theta * self.dt + (self.nodes * v_init) * self.dt
        sq_rho = np.sqrt(1.0 - self.rho ** 2)
        for i in range(self.num_time_step):
            sq_v_now = np.squeeze(np.sqrt(np.maximum(v_now @ self.weights[:, np.newaxis], np.array([0.0]))))
            x_now += self.rate * self.dt + sq_v_now * (self.rho * dw_sample[:, 1, i] + sq_rho * dw_sample[:, 0, i]) \
                    - 0.5 * sq_v_now ** 2 * self.dt
            v_now = (v_now + self.nu * (sq_v_now * dw_sample[:, 1, i])[:, np.newaxis] + b) @ A_inv

            s_sample[:, i+1] = np.exp(x_now)
            v_factors[:, :, i+1] = v_now
            v_sample[:, i+1] = np.squeeze(np.maximum(v_now @ self.weights[:, np.newaxis], np.array([0.0])))

        return dw_sample, s_sample, v_sample, v_factors
    
    def lsmc(self, num_sample, degree):
        _, _, u_sample, v_factors = self.sample(num_sample)
        a_sample = self.weights @ np.diag(self.nodes) @ v_factors
        cond_exp = np.zeros(self.num_time_step + 1)
        cond_exp[0] = a_sample[0, 0]

        # Apply least square methods
        for i in range(1, self.num_time_step + 1):
            regression = np.polyfit(u_sample[:, i], a_sample[:, i], degree)
            cond_exp[i-1] = np.mean(np.polyval(regression, u_sample[:, i]))
        
        return cond_exp

    



