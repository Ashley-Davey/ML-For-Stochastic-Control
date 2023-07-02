import numpy as np
from numpy.linalg import inv
from scipy.sparse import random
import analytical
from date import log_config
from numpy import log, exp, power, dot
from scipy.integrate import odeint 
from scipy.optimize import minimize


class Config:
    def __init__(self, **kwargs):
        self.initialise_parameters()
        self.make_changes(**kwargs)
        self.initialise_others()
        self.solve()
        self.rate_start = {
            "primal": [0.01, 0.01],
            "dual": [0.01, 0.01],
            "smp": [0.01, 0.01],
            "pde": [0.01, 0.0],
            "bruteprimal": [0.0, 0.01],
            "brutedual": [0.0, 0.01],
            "hybrid": [0.01, 0.01],
        }
        self.max_iter = 1000
        self.display_steps =  round(self.max_iter) / 5
        self.decay_steps = round(self.max_iter ) / 2
        self.tol = 1e-9

    def initialise_parameters(self):
        self.layers = 2
        self.extra = 20
        self.final_size = None
        self.sample_size = 1
        self.sample_size_control = 100000
        self.final = 1
        self.repeats = 1

        self.T = 1.0
        self.N = 10
        self.d = 1
        self.k = 1
        self.space = 'whole'  #'whole' ~ R^m, 'cone' ~ [0,\infty]^m, or 'ball' ~ B(0,1) or 'zero' ~ {0}
        self.m = 1
        self.X0 = 1.0 
        self.alpha = 0.0
        self.batch_size = 64
        self.editables = ['layers', 'extra', 'T', 'N', 'k', 'd', 'm', 'space', 'X0', 'alpha', 'batch_size', 'sample_size']

    def make_changes(self, **kwargs):
        for name, value in kwargs.items():
            if name in self.editables:
                print(f"Changed {name} from {getattr(self,name)} to {value}")
                setattr(self, name, value)
            else:
                print(f"Did not change {name}")

    def initialise_others(self):
        self.h = self.T / self.N
        self.sqrth = np.sqrt(self.h)
        self.Xinit = self.X0 * np.ones(self.d)
        if self.sample_size > 1:
            print(f'Warning: sample size is {self.sample_size}' )


    def dump(self):
        log_config(f"Config for {type(self).__name__[:-6]}")
        for key, val in self.__dict__.items():
            log_config(f"{key}: {val}")

    def solve(self):
        try:
            self.solution = getattr(analytical, f'Solve{self.name}')(self)
        except Exception as e:
            print(e)
            self.solution = analytical.Solution(self)


class UtilityConfig(Config):
    def initialise_parameters(self):
       super().initialise_parameters()
       self.c = False
       self.a = 0.0
       self.eta = 0.0
       self.delta = 0.0
       self._r = 0.05
       self._sigma = 0.2
       self._mu = 0.06
       self.editables +=  ['pi', 'c', 'a', 'eta', '_mu', '_r', '_sigma', 'd', 'delta']


    def initialise_others(self):
        super().initialise_others()
        self.gamma = (self.eta != 0.0)
        self.alpha = self.delta + self.eta
        self.mu = lambda t : self._mu * np.ones(self.k)
        self.r = lambda t : self._r
        seed = 12345
        np.random.seed(seed)
        if self.space in ["whole", "zero"]:
            self._mu =  np.ones(self.k) * 0.06
            self.mu = lambda t: self._mu
            self._sigma = np.array(random(self.k, self.k, density = 0.1, random_state = seed)*0.05  + np.eye(self.k) * 0.2)
            self.sigma = lambda t: self._sigma
        elif self.space == "cone":
            self._sigma = 0.2 + np.eye(self.k) * 0.2
            self.sigma = lambda t: self._sigma
            self._freq = np.ones(self.k) * np.pi
            self._amp = np.ones(self.k)   * 0.2
            self._off = np.random.rand(self.k) * 2 * np.pi + 3
            self.mu = lambda t:  0.05 + np.sin(self._freq * t + self._off) * self._amp
        elif self.space == "ball":
            self._off = np.random.rand(self.k) * 2 * np.pi
            self.mu = lambda t: 0.07 * np.ones(self.k)
            self.sigma = lambda t: np.diag((4 + np.sin(2 * t * np.pi + self._off)) / 10)
        self.m = self.k  + (1 if self.c else 0)
        self.m_dual = self.k  + (1 if self.gamma else 0)
        self.theta = lambda t: (inv(self.sigma(t)) @ (self.mu(t) - self.r(t)))
        self.theta_mod = lambda t: dot(self.theta(t), self.theta(t))


class PowConfig(UtilityConfig):
    def initialise_parameters(self):
        super().initialise_parameters()
        self.name = "Pow"
        self.p =  0.5

    def initialise_others(self):
        super().initialise_others()
        self.q = self.p / (self.p - 1)
        self.H = lambda t : (1 - exp(-self.r(t) * (self.T - t))) / self.r(t)
        if self.c:
            k = lambda t : (1 - self.q) * self.delta + self.q * self.r(t) + 0.5 * self.theta_mod(t) * (1 - self.q) * self.q
            self.G = lambda t : ( 1 - 1 / k(t)) *  exp(-k(t) * (self.T - t))  + 1 / k(t)
            self.F = lambda t : power(self.G(t), 1 - self.p)
        else:
            k = lambda t : self.delta - self.p * self.r(t) + 0.5 * self.theta_mod(t) * self.q
            self.F = lambda t : exp(-k(t) * (self.T - t))
            self.G = lambda t: power(self.F(t), 1 / (1 - self.p))

    def I(self, x):
        return np.power(x, 1 / (self.p - 1))

    def U(self, x):
        return np.power(x, self.p) / self.p

class YaariConfig(UtilityConfig):
    def initialise_parameters(self):
        super().initialise_parameters()
        self.name = "Yaari"
        self.H =  1.5
        assert self.k == 1

    def initialise_others(self):
        super().initialise_others()
        assert self.Xinit[0] < self.H

    def I(self, x):
        raise ValueError("I does not exist for this problem")

    def U(self, x):
        return np.minimum(x, self.H)

class QuadConfig(UtilityConfig):
    def initialise_parameters(self):
        super().initialise_parameters()
        self.name = "Quad"
        self.C = -1.0
        self.H =  2.0
        self.beta = 0.55 #alpha was taken
        self.editables += ['C', 'H', 'beta']
        assert self.k == 1
        
    def initialise_others(self):
        super().initialise_others()
        
    def I(self, x):
        raise ValueError("I does not exist for this problem")

    def U(self, x):
        return np.where(
            x < self.beta * self.H,
            0.5 * self.C * x ** 2 - self.C * self.H * x,
            self.C * self.beta * self.H ** 2 * (0.5 * self.beta - 1)
            )


class LogConfig(UtilityConfig):
    def initialise_parameters(self):
        super().initialise_parameters()
        self.name = "Log"
        self.X0 = [1.0]

    def initialise_others(self):
        super().initialise_others()

    def I(self, x):
        return 1 / x

    def U(self, x):
        return np.log(x)

class HConfig(UtilityConfig):
    def initialise_parameters(self):
        super().initialise_parameters()
        self.name = "H"

    def initialise_others(self):
        super().initialise_others()
        
        
class LQConfig(Config):
    def initialise_parameters(self):
        super().initialise_parameters()
        self.name = "LQ"
        self.space = 'whole' #whole or cone ([0,inf))
        self.r = 1.0
        self.times = 1000
        
    def value(self, t, x):
        #gets value, given intial matrix a
        return 0.5 * np.dot(self.a_hat[self.part(t)] @ x, x)

    def part(self, t):
        #get element of partition for t
        return max(min(round(t * self.times / self.T ) , self.times - 1), 0)
        
    def initialise_others(self):
        super().initialise_others()
        assert self.k == 1
        np.random.seed(1)
        
        _A = (np.random.rand(self.d,self.d) - 0.5)  / (  2 * max(self.d, 5) ** 1.5)
        _B = (np.random.rand(self.d,self.m) - 0.5)  / (  2 * max(self.d, 5) ** 1.5)
        _C = (np.random.rand(self.d,self.d) - 0.5)  
        _D = (np.random.rand(self.d,self.m) - 0.5)  / (  2 * max(self.d, 5) ** 1.5)  
        
        # _A = np.ones((self.d,self.d) ) / 10
        # _B = np.ones((self.d,self.m) ) / 10
        # _C = np.ones((self.d,self.d) ) / 10
        # _D = np.ones((self.d,self.m) ) / 10
        
        self.A = lambda t: _A * np.sin(t *  2 * np.pi) + 0.01
        self.B = lambda t: _B * np.cos(t *  2 * np.pi) + 0.01
        self.C = lambda t: _C * np.cos(t *  2 * np.pi) + 0.01
        self.D = lambda t: _D * np.sin(t *  2 * np.pi) + 0.01
        
        _Q =  (np.random.rand(self.d,self.d) - 0.5) / ( max(self.d, 5)  ** 1.5) 
        # _Q =  np.ones((self.d,self.d)) / 20
        self.Q = lambda t: (_Q + _Q.T) * np.cos(t * 2 * np.pi)
        
        self.R = lambda t: np.eye(self.m) * self.r

        #S is m by n
        _S = np.random.rand(self.m,self.d) / (  max(self.d, 5)  ** 2)
        # _S = np.ones((self.m,self.d) ) / 10
        self.S = lambda t: _S
        
        _G = np.ones((self.d,self.d)) / 20
        self.G = _G + _G.T
        
        
        _gamma =  (np.random.rand(self.d) - 0.5) 
        _sigma =  (np.random.rand(self.d) - 0.5)
        
        _q     =  (np.random.rand(self.d) - 0.5) 
        _p     =  (np.random.rand(self.m) - 0.5)
        
        _g     =  (np.random.rand(self.d) - 0.5)

        self.gamma = lambda t: _gamma
        
        self.sigma = lambda t: _sigma
        self.q     = lambda t: _q
        
        self.p     = lambda t: _p
        self.g     = _g
        
        if self.space == 'whole':
            
            def fun(t, a):
                a_temp = a.reshape(self.d, self.d)
                K = self.D(t).T @ (a_temp @ self.D(t)) + self.R(t)
                L = self.B(t).T @ a_temp + self.D(t).T @ (a_temp @ self.C(t)) + self.S(t)
        
                ret  = (
                    a_temp @ self.A(t)
                    + self.A(t).T @ a_temp
                    + self.C(t).T @ (a_temp @ self.C(t))
                    + self.Q(t)
                    - L.T @ (np.linalg.inv(K) @ L)
                    )
                
                return -1 * ret.reshape(self.d ** 2)
            
        
            
            self.a_hat = odeint(
                fun,
                self.G.reshape(self.d ** 2),
                np.linspace(self.T, 0, self.times),
                tfirst = True
                ).reshape(self.times, self.d, self.d)[::-1]
            
            
            times = np.linspace(0, self.T, self.times)
            
            BT = np.array([self.B(t).T for t in times])
            C = np.array([self.C(t) for t in times])
            D = np.array([self.D(t) for t in times])
            DT = np.array([self.D(t).T for t in times])
            R = np.array([self.R(t) for t in times])
            S = np.array([self.S(t) for t in times])
    
                        
            K = DT @ (self.a_hat @ D) + R
            L = BT @ self.a_hat + DT @ (self.a_hat @ C) + S            
            
            def fun(t, b):
                n = self.part(t)
                a = self.a_hat[n]
                
                M = self.B(t).T @ b + self.D(t).T @ (a @ self.sigma(t)) + self.p(t)
                
                ret  = (
                    self.A(t).T @ b
                    + a @ self.gamma(t)
                    + self.C(t).T @ (a @ self.sigma(t))
                    + self.q(t)
                    - L[n].T @ (np.linalg.inv(K[n]) @ M)
                    )
                
                return -1 * ret.reshape(self.d)
            
        
            
            self.b_hat = odeint(
                fun,
                self.g,
                np.linspace(self.T, 0, self.times),
                tfirst = True
                ).reshape(self.times, self.d)[::-1]    
            
            sigma = np.array([self.sigma(t) for t in times])
            p = np.array([self.p(t) for t in times])
            
            M1 = np.squeeze(BT @ np.expand_dims(self.b_hat, -1), -1)
            M2 = np.squeeze(DT @ (self.a_hat @ np.expand_dims(sigma, -1)), -1) + p
            
            M = M1 + M2
            
            
            def fun(t, xi):
                n = self.part(t)
                a = self.a_hat[n]
                b = self.b_hat[n]
                
                
                ret  = (
                    b @ self.gamma(t)
                    + 0.5 * self.sigma(t) @ (a @ self.sigma(t))
                    - 0.5 * M[n].T @ (np.linalg.inv(K[n]) @ M[n])
                    )
                
                return -1 * ret
            
        
            
            self.xi_hat = odeint(
                fun,
                0,
                np.linspace(self.T, 0, self.times),
                tfirst = True
                ).reshape(self.times)[::-1]          
            
            
            self.alpha_hat = - np.linalg.inv(K) @ L
        
            
            self.beta_hat  = - np.squeeze(np.linalg.inv(K) @  np.expand_dims(M, -1), -1)
            
                        

            
        elif self.space == 'cone':
            assert self.d == 1
            assert np.isclose(
                0,
                np.abs(self.gamma(0)) 
                + np.abs(self.sigma(0)) 
                + np.abs(self.p(0)) 
                + np.abs(self.q(0)) 
                + np.abs(self.g) 
                )
            
            def f_hat(t, k, a):
                term1 = np.dot(k, (a * self.D(t).T @ self.D(t) + self.R(t)/ 2) @ k)
                term2 = (2 * (a * self.C(t) @ self.D(t) + a * self.B(t) + self.S(t).T / 2) @ k)[0]
                return term1 + term2
            
            def f_bar(t, k, a):
                term1 = np.dot(k, (a * self.D(t).T @ self.D(t) + self.R(t) / 2) @ k)
                term2 = (2 * (a * self.C(t) @ self.D(t) + a * self.B(t) + self.S(t).T / 2) @ k)[0]
                return term1 - term2
            
            bounds = [(0, np.inf)] * self.m
            
            def fun_hat(t, a):
                f_hat_opt = lambda k: f_hat(t, k, a)
                
                f_term = minimize(f_hat_opt, np.zeros(self.m), bounds = bounds).fun
                
                return -1 * (
                    a * self.C(t)[0,0] ** 2 +
                    2 * a * self.A(t)[0,0] +
                    self.Q(t)[0,0] / 2 +
                    f_term
                    )
            
            def fun_bar(t, a):
                f_bar_opt = lambda k: f_bar(t, k, a)
                
                f_term = minimize(f_bar_opt, np.zeros(self.m), bounds = bounds ).fun
                
                return -1 * (
                    a * self.C(t)[0,0] ** 2 +
                    2 * a * self.A(t)[0,0] +
                    self.Q(t)[0,0] / 2 +
                    f_term
                    )
                
    
            self.a_hat = odeint(
                fun_hat,
                self.G[0,0] / 2,
                np.linspace(self.T, 0, self.times),
                tfirst = True
                )[::-1]
            
            self.a_bar = odeint(
                fun_bar,
                self.G[0,0] / 2,
                np.linspace(self.T, 0, self.times),
                tfirst = True
                )[::-1]     
            
            self.K_hat = []
            self.K_bar = []

            
            for t in np.linspace(0, self.T, self.times):
            
                f_hat_opt = lambda k: f_hat(t, k, self.a_hat[self.part(t)])
                K = minimize(f_hat_opt, np.zeros(self.m), bounds = bounds ).x.reshape(self.m, 1)
                self.K_hat.append(K)
                
                f_bar_opt = lambda k: f_bar(t, k, self.a_bar[self.part(t)])
                K = minimize(f_bar_opt, np.zeros(self.m), bounds = bounds ).x.reshape(self.m, 1)
                self.K_bar.append(K)
                
                
        





def get_config(name, **kwargs):
    try:
        return globals()[name + "Config"](**kwargs)

    except KeyError:
        raise KeyError("Config for the required problem not found.")
