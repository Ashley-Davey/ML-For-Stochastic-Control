# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:46:38 2019

@author: Ashley
"""

import numpy as np
from autograd.numpy.linalg import inv
from autograd.numpy import power, exp, dot, log, sqrt, cos, sin
from autograd import grad, hessian
import time
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as normal
from scipy.stats import norm
from scipy.integrate import quad, odeint

pdf = norm.pdf
cdf = norm.cdf
ppf = norm.ppf


class Solution():
    def __init__(self, config):
        self.config = config
        self.soln = np.squeeze(self.u(0, config.Xinit))
    


    def u(self, t, x):
        return None

    def v(self,t,y):
        result = minimize(lambda x : -1 * self.u(t, x) + x * y, [0])
        ret = -1 * result.fun if result.success else result.fun[0]
        return ret

    def Du(self, t, x):
        if self.u is None:
            return None
        _u = lambda y: self.u(t, y)
        grad_u = grad(_u)
        return grad_u(x)

    def DDu(self, t, x):
        if self.u is None:
            return None
        _u = lambda y: self.u(t, y)
        ret = hessian(_u)(x)
        if len(np.array(x).shape) == 0:
            ret = ret.reshape((1,1))
        return ret

    def V1(self, t, w):
        X = self.X(t, w)
        return self.u(t, X)

    def V(self,t, w):
        return self.V1(t, w)

    def Z1(self, t, w):
        X = self.X(t, w)
        return self.Du(t, X)

    def Gamma1(self, t, w):
        X = self.X(t, w)
        return self.DDu(t, X)

    def primal_control(self, t, w):
        X = self.X(t, w)
        return self.control(t, X)

    def X(self, t, W):
        raise NotImplementedError('No X')

    def control(self, t, x):
        raise NotImplementedError('No control')


def SolvePow(config):
    if config.space != 'whole':
        return Solution(config)
    elif config.a > 0 and config.eta > 0:
        return Solution(config)
    else:
        return PowSolution(config)

def SolveLog(config):
    if config.space != 'whole':
        return Solution(config)
    elif config.a > 0 and config.eta > 0:
        return Solution(config)
    else:
        return LogSolution(config)

def SolveH(config):
    if config.space != 'whole':
        return Solution(config)
    elif config.a > 0 and config.eta > 0:
        return Solution(config)
    else:
        return HSolution(config)

def SolveYaari(config):
    if config.space != 'whole':
        return Solution(config)
    else:
        return YaariSolution(config)

def SolveQuad(config):
    if config.space != 'whole':
        return Solution(config)
    else:
        return QuadSolution(config)


class UtilitySolution(Solution):
    def control(self, t, x):
        pi = dot(inv(self.config.sigma(t)).T, self.config.theta(t)) * self.Du(t, x) / (x * self.DDu(t, x))
        if self.config.c:
            c = self.config.I(self.Du(t, x))
            return np.concatenate((pi, c))
        else:
            return pi

    def d_control(self, t, y):
        raise NotImplementedError('No Dual Control')

    def dual_control(self, t, w):
        Y = self.Y(t, w)
        return self.d_control(t, Y)

    def Y(self, t, w):
        return self.Z1(t, w)
    
    def theta_hat(self, t):
        return self.primal_control(t, None)

class PowSolution(UtilitySolution):
    def u(self, t, x):
        return self.config.U(x + self.config.a * self.config.H(t)) * self.config.F(t)


    def X(self, t, w):
        if self.config.c or self.config.a > 0:
            raise ValueError("No solution path for consumption or income (yet)")
        drift = lambda t: self.config.r(t) + (1 - 2 * self.config.p) * self.config.theta_mod(t) / (2 * (1 - self.config.p) ** 2)
        vol = lambda t: self.config.theta(t) / (1 - self.config.p)

        drift_term = quad(drift, 0, t)[0]

        n = int(t / self.config.h)
        if n == 0:
            vol_term = 0
        else:
            vol_term = np.sum([np.dot(vol(m * self.config.h), (w[m+1] - w[m])) for m in range(n)])

        return self.config.Xinit[0] * exp( drift_term + vol_term)

    def primal_control(self, t, w):
        return dot(inv(self.config.sigma(t)), self.config.theta(t)) / (1 - self.config.p)

    def dual_control(self, t, y):
        return np.zeros(self.config.m) ############################### only unconstrained


class HSolution(UtilitySolution):

    def y(self,t,x):
        r = self.config.r(0) ######################## only constant coefficients
        theta_mod = self.config.theta_mod(0)   ############# only whole space
        T = self.config.T

        sum1 = exp((r + theta_mod) * (T - t))
        sum2 = sqrt(exp(2 * (r + theta_mod) * (T - t)) + 4 * x * exp(3 * (r + 2 * theta_mod) * (T - t)))

        return sqrt(sum1 + sum2) / sqrt(2 * x)

    def u(self, t, x):
        y = self.y(t, x)
        r = self.config.r(0)
        theta_mod = self.config.theta_mod(0)
        T = self.config.T

        return (2 / 3) * (exp((r + theta_mod) * (T - t)) / y  + 2 * x * y)

    def a_1(self):
        y = self.y(0, self.config.Xinit[0])
        r = self.config.r(0)
        theta_mod = self.config.theta_mod(0)
        T = self.config.T

        return exp(3 * (r + 2 * theta_mod) * T) / (y ** 4)

    def a_2(self):
        y = self.y(0, self.config.Xinit[0])
        r = self.config.r(0)
        theta_mod = self.config.theta_mod(0)
        T = self.config.T

        return exp((r + theta_mod) * T) / (y ** 2)

    def S_1(self, t, w):
        r = self.config.r(0)
        theta_mod = self.config.theta_mod(0)
        theta = self.config.theta(0)

        return exp((r - 4 * theta_mod) * t + 4 * dot(theta, w[-1]))

    def S_2(self, t, w):
        r = self.config.r(0)
        theta = self.config.theta(0)
        return exp(r * t + 2 * dot(theta, w[-1]))



    def X(self, t, w):
        if self.config.c or self.config.a > 0:
            raise ValueError("No solution path for consumption or income (yet)")

        return self.a_1() * self.S_1(t, w) + self.a_2() * self.S_2(t, w)

    def Q(self, t, w):
        theta = self.config.theta(0)
        return (4 * self.a_1() * self.S_1(t, w) + 2 * self.a_2() * self.S_2(t, w)) * theta

    def primal_control(self, t, w):
        sigma = self.config.sigma(0)

        return dot( inv(sigma).T, self.Q(t, w) / self.X(t, w))





    def dual_control(self, t, y):
        return np.zeros(self.config.m) ################ only unconstrained



class LogSolution(UtilitySolution):

    def y(self, t, x):
        return 1 / x

    def v(self, t, y):
        r = self.config.r(0)
        theta_mod = self.config.theta_mod(0)
        T = self.config.T
        return - 1 - log(y) + (r + theta_mod / 2) * (T - t)

    def u(self, t, x):
        y = self.y(t,x)
        return self.v(t, y) + x * y

    def Y(self, t, w):
        r = self.config.r(0)
        theta = self.config.theta(0)
        theta_mod = self.config.theta_mod(0)
        y = self.y(0, self.config.Xinit[0])

        return y * exp( (- r - theta_mod / 2) * t - np.dot(theta, w[-1]))




    def X(self, t, w):
        if self.config.c or self.config.a > 0:
            raise ValueError("No solution path for consumption or income (yet)")

        return 1 / self.Y(t, w)



    def dual_control(self, t, y):
        return np.zeros(self.config.m) ################ only unconstrained

    def primal_control(self, t, w):
        sigma = self.config.sigma(0)
        theta = self.config.theta(0)

        return dot(inv(sigma.T), theta)






class YaariSolution(UtilitySolution):
    def phi_x(self, t, x):
        return ppf(x * exp(self.config.r(0) * (self.config.T - t)) / self.config.H)


    def u(self, t, x):
        return self.config.H * cdf(self.phi_x(t, x) + self.config.theta(0)[0] * sqrt(self.config.T - t))

    def Du(self, t, x):
        exp_term = exp(self.config.r(0) * (self.config.T - t))
        div_term = pdf(self.phi_x(t, x))
        mult_term = pdf(
            self.phi_x(t, x) + self.config.theta(0)[0] * sqrt(self.config.T - t)
            )

        return exp_term * mult_term / div_term

    def DDu(self, t, x):
        theta_term = self.config.theta(0)[0] * sqrt(self.config.T-t)
        numerator = -exp(2 * self.config.r(0) * (self.config.T - t)) * theta_term* pdf(self.phi_x(t, x) + theta_term)


        denominator = self.config.H * pdf(self.phi_x(t,x)) ** 2
        return numerator / denominator

    def X(self, t, w):
        if self.config.c or self.config.a > 0:
            raise ValueError("No solution path for consumption or income (yet)")

        H = self.config.H
        r = self.config.r(0)
        theta = self.config.theta(0)
        Z_0 = ppf(self.config.Xinit[0] * exp(r * self.config.T) / H)
        tau = self.config.T - t

        if t == self.config.T:
            if np.sum(sqrt(self.config.T) * Z_0 +  theta * self.config.T + w[-1]) > 0:
                return np.array([H])
            else:
                return np.array([0])
        else:
            return H * exp(-r * tau) * cdf((sqrt(self.config.T) * Z_0 +  theta * t + w[-1]) / sqrt(tau))

    def Z1(self, t, w):
        H = self.config.H
        r = self.config.r(0)
        theta = self.config.theta(0)
        theta_mod = self.config.theta_mod(0)
        Z_0 = ppf(self.config.Xinit[0] * exp(r * self.config.T) / H)
        T = self.config.T

        Y_0 = exp(-theta * sqrt(T) * Z_0 + (r - theta_mod / 2) * T )

        return np.dot(Y_0, exp((-r - theta_mod / 2) * t - np.dot(theta, w[-1])))

    def primal_control(self, t, w):
        X = self.X(t, w)
        H = self.config.H
        r = self.config.r(0)
        sigma = self.config.sigma(0)
        tau = self.config.T - t

        mult1 = dot(inv(sigma), 1 / X) * H * exp(-r * tau) / sqrt(tau)

        mult2 = pdf(ppf(X * exp(r * tau) / H))

        return mult1*mult2



    def dual_control(self, t, y):
        return np.zeros(self.config.m)


class QuadSolution(UtilitySolution):
    def __init__(self, config):
        super().__init__(config)
        self.y0 = self.find_y()




    def find_y(self):
        f = lambda y : self.v(0, y) + self.config.Xinit[0] * y
        return minimize(f, 1.0, bounds = [(1e-8, np.inf)]).x[0]


    def k(self, t, y):
        tau = self.config.T - t
        theta = self.config.theta(0)[0]
        c = self.config.C
        H = self.config.H
        alpha = self.config.beta
        r = self.config.r(0)


        if t == self.config.T:
            return np.inf * np.sign(log(y) - log(c * H * (alpha - 1)))
        else:
            return (log(y) - log(c * H * (alpha - 1)) - (r + theta ** 2 / 2) * tau) / (theta * sqrt(tau))

    def v(self, t, y):
        tau = self.config.T - t
        theta = self.config.theta(0)[0]
        c = self.config.C
        H = self.config.H
        alpha = self.config.beta
        r = self.config.r(0)

        return (
            cdf(- self.k(t,y)) * alpha * c * H ** 2 * (alpha - 2) / 2 +
            - cdf(-self.k(t,y) - theta * sqrt(tau)) * alpha * H * y * exp(-r * tau) +
            - cdf(self.k(t,y)) * c * H ** 2 / 2 +
            - cdf(self.k(t,y) + 2 * theta * sqrt(tau)) * y ** 2 * exp((-2 * r + theta ** 2) * tau) / (2 * c) +
            - cdf(self.k(t,y) + theta * sqrt(tau)) * H * y * exp(-r * tau)
            )


    def dv(self, t, y):
        tau = self.config.T - t
        theta = self.config.theta(0)[0]
        c = self.config.C
        H = self.config.H
        alpha = self.config.beta
        r = self.config.r(0)

        return (
            - alpha * H * exp(-r * tau)
            - cdf(self.k(t,y) + theta * sqrt(tau)) * (1 - alpha) * H * exp(-r * tau)
            - cdf(self.k(t,y) + 2 * theta * sqrt(tau)) * y * exp((-2 * r + theta ** 2) * tau) / c
            )

    def u(self, t,x):
        f = lambda y : self.v(t, y) + x * y

        return minimize(f, 1.0, bounds = [(1e-8, np.inf)]).fun[0]

    def Y(self, t, W):
        theta = self.config.theta(0)[0]
        r = self.config.r(0)

        return self.y0 * exp((- r - theta ** 2 / 2 ) * t - theta * W[-1])



    def X(self, t, W):
        return -1 * self.dv(t, self.Y(t, W))

    def Z1(self, t, W):
        return self.Y(t, W)

    def ddv(self, t, y):
        tau = self.config.T - t
        theta = self.config.theta(0)[0]
        c = self.config.C
        r = self.config.r(0)

        return -cdf (self.k(t,y) + 2 * theta * sqrt(tau)) * exp((-2 * r + theta ** 2) * tau) / c

    def Gamma1(self, t, W):
        return -1 / self.ddv(t, self.Y(t, W))

    def primal_control(self, t, W):
        theta = self.config.theta(0)[0]
        sigma = self.config.sigma(0)[0, 0]

        return -1 * (theta / sigma) * self.Z1(t, W) / self.Gamma1(t, W)

    def dual_control(self, t, W):
        return np.zeros(self.config.k)

class LQSolution(Solution):
    def u(self, t, x):
        return (
            0.5 * np.dot(self.config.a_hat[self.config.part(t)] @ x, x)
            + self.config.b_hat[self.config.part(t)] @ x
            + self.config.xi_hat[self.config.part(t)]
            )
    
    def Du(self, t, x):
        return  (
            self.config.a_hat[self.config.part(t)] @ x 
            + self.config.b_hat[self.config.part(t)]
            )
    
    def DDu(self, t, x):
        return self.config.a_hat[self.config.part(t)]
    
    def X(self, t, W):
        _X = self.config.X0 * np.ones(self.config.d)
        
        for s in range(int(t / self.config.h)):
            b   = (
                self.config.A(s * self.config.h) @ _X 
                + self.config.B(s * self.config.h) @ self.control(s * self.config.h, _X)
                + self.config.gamma(s * self.config.h)
                )
            sig = (
                self.config.C(s * self.config.h) @ _X 
                + self.config.D(s * self.config.h) @ self.control(s * self.config.h, _X)
                + self.config.sigma(s * self.config.h)
                )
            dW = W[s + 1] - W[s]
            _X += b * self.config.h + sig * dW
        return _X
        
        

    def control(self, t, x):
        if len(x) == 1:
            return (
                self.config.alpha_hat[self.config.part(t)] @ x 
                + self.config.beta_hat[self.config.part(t)]
                )
        else:
            return (
                (self.config.alpha_hat[self.config.part(t)][np.newaxis, :, :] @ x[:, :, np.newaxis ]).squeeze(-1)
                + self.config.beta_hat[self.config.part(t)]
                )

class LQConeSolution(Solution):
    def u(self, t, x):
        if x[0] >= 0:
            g = self.config.a_hat[self.config.part(t)]
        else:
            g = self.config.a_bar[self.config.part(t)]
        return g * x[0] ** 2
    
    def Du(self, t, x):
        if x[0] >= 0:
            g = self.config.a_hat[self.config.part(t)]
        else:
            g = self.config.a_bar[self.config.part(t)]
        return 2 * g * x[0] 


    def DDu(self, t, x):
        if x[0] >= 0:
            g = self.config.a_hat[self.config.part(t)]
        else:
            g = self.config.a_bar[self.config.part(t)]
        return 2 * g 

    
    def X(self, t, W):
        _X = self.config.X0 * np.ones(self.config.d)
        
        for s in range(int(t / self.config.h)):
            b   = self.config.A(s * self.config.h) @ _X + self.config.B(s * self.config.h) @ self.control(s * self.config.h, _X)
            sig = self.config.C(s * self.config.h) @ _X + self.config.D(s * self.config.h) @ self.control(s * self.config.h, _X)
            dW = W[s + 1] - W[s]
            _X += b * self.config.h + sig * dW
        return _X
        
        

    def control(self, t, x):
        K_hat = self.config.K_hat[self.config.part(t)]
        K_bar = -1 * self.config.K_bar[self.config.part(t)]
        
        
        if len(x) == 1:
            return  np.where(x > 0, K_hat @ x, K_bar @ x)
        else:
            return  np.where(x > 0,
                             (K_hat[np.newaxis, :, :] @ x[:, :, np.newaxis ]).squeeze(-1),
                             (K_bar[np.newaxis, :, :] @ x[:, :, np.newaxis ]).squeeze(-1)
                             )
    



def SolveLQ(config):
    if config.space == 'whole':
        return LQSolution(config)
    elif config.space == 'cone':
        return LQConeSolution(config)
    else:
        return Solution(config)

















