import tensorflow as tf
import numpy as np
from numpy import exp, sqrt, power, log
from autograd.numpy.linalg import inv
from NN import net
from scipy.stats import expon, multivariate_normal as normal

def b_mult(A, B): #batch matrix x vector multiplication
    assert len(A.shape) - 1 == len(B.shape) == 2
    return tf.squeeze(A @ tf.expand_dims(B, -1), -1)

def b_dot(A,B): #batch vector dot product
    assert len(A.shape) == len(B.shape) == 2
    return tf.reduce_sum(A * B, 1, keepdims = True)


class Method:
    def __init__(self, config, names, graph, monte):
        self.config = config
        self.graph = graph
        self.monte = monte
        self.h = self.config.h
        self.sqrth = self.config.sqrth
        self.N = self.config.N
        self.d = self.config.d
        self.m = self.config.m
        self.k = self.config.k
        self.T = self.config.T
        self.names = names
        self.randoms = ['dW']
        self.dW = tf.placeholder(tf.float64, [None, self.k, self.N], name="dW")
        self.size = tf.shape(self.dW)[0]
        self.allones = tf.ones(
            shape=tf.stack([self.size, 1]), dtype=tf.float64, name="MatrixOfOnes"
        )
        self._extra_train_ops_control = []
        self._extra_train_ops_bsde = []

        self.process_keys = {'common': ['W']}

        for name in self.names:
            try:
                getattr(self, f'load_{name}')()
            except KeyError as e:
                print(e)
                self.names.remove(name)

    def initialise_processes(self):
        self.processes = {}
        for name in self.process_keys.keys():
                for key in self.process_keys[name]:
                    if key != 'V':
                        var = getattr(self, f'{key}_init')(name)
                        try:
                            getattr(self, f'{key}_0')[name] = var
                        except AttributeError:
                            setattr(self, f'{key}_0', {})
                            getattr(self, f'{key}_0')[name] = var
                        var =  tf.ones(shape=[self.size] + var.shape.as_list()[1:], dtype=tf.float64) * var

                        try:
                            self.processes[key][name] = {0 : var}
                        except KeyError:
                            self.processes[key] = {}
                            self.processes[key][name] = {0 : var}

    def update_processes(self, t): # t = 0, ... ,N-1
        for name in self.process_keys.keys():
            for key in self.process_keys[name]:
                if key != 'V':
                    update = getattr(self, f'{key}_update')(t, name)
                    if update is not None:
                        self.processes[key][name][t+1] = update


    def makeBackwardsProcesses(self):
        for name in self.process_keys.keys():
            if 'V' in self.process_keys[name]:
                key = 'V'
                var = getattr(self, f'{key}_terminal')(name)
                try:
                    self.processes[key][name] = {self.N : var}
                except KeyError:
                    self.processes[key] = {}
                    self.processes[key][name] = {self.N : var}
                for t in range(self.N - 1, -1, -1):
                    update = getattr(self, f'{key}_update')(t, name)
                    if update is not None:
                        self.processes[key][name][t] = update

    def fixProcesses(self):
        pass


    def W_init(self, name):
        return tf.zeros([1, self.k], dtype=tf.float64)

    def W_update(self, t, name):
        return self.processes['W'][name][t] + self.sqrth * self.dW[:, :, t]

    def get_vars(self):
        self.variables = {}
        for name in self.names:
            self.variables[name] = getattr(self, f'get_vars_{name}')()

    def get_losses(self):
        self.loss_bsde = {}
        self.loss_control = {}
        self.loss_y = {}
        for name in self.names:
            self.loss_bsde[name], self.loss_control[name], self.loss_y[name]  = getattr(self, f'get_losses_{name}')()


    def load_primal(self):
        self.process_keys['primal'] = ['X', 'Z1', 'V1', 'Gamma1', 'primal_control', 'J1']
        self.n_neuron_Gamma = (
            [self.d] + [self.d + self.config.extra] * self.config.layers + [self.d ** 2]
        )
        self.n_neuron_primal_control = (
                [self.d] + [self.d + self.config.extra] * self.config.layers + [self.m]
        )

    def X_init(self, name):
        X_0 = self.config.Xinit
        return tf.ones([1, self.d], dtype=tf.float64) * X_0

    def X_update(self, t, name):
        X = self.processes['X'][name]
        pi = self.processes['primal_control'][name]
        return ( X[t] +
            self.drift(t * self.h, X[t], pi[t]) * self.h
            + b_mult(self.vol(t * self.h, X[t], pi[t]), self.dW[:, :, t]) * self.sqrth
            )

    def drift(self, t, X, pi):
        raise NotImplementedError

    def vol(self, t, X, pi):
        raise NotImplementedError

    def Z1_init(self, name):
        var = tf.Variable(
                tf.random_uniform([1, self.d], minval=-0.1, maxval=0.1, dtype=tf.float64),
                name=f"Z1_0_{name}",
            )

        return var


    def Z1_update(self, t, name):
        X = self.processes['X'][name]
        pi = self.processes['primal_control'][name]
        Z = self.processes['Z1'][name]
        Gamma = self.processes['Gamma1'][name]

        Q = Gamma[t] @ self.vol(t * self.h, X[t], pi[t])
        drift = self.config.alpha * Z[t] - self.grad_H(t, X[t], pi[t], Z[t], Q)
        return ( Z[t] +
            drift * self.h
            + b_mult(Q, self.dW[:, :, t]) * self.sqrth
        )

    def grad_H(self, t, X, pi, Z, Q):
        volT = tf.linalg.transpose(self.vol(t * self.h, X, pi))
        drift = self.drift(t * self.h, X, pi)
        H = (
            b_dot(drift, Z) +
            tf.reshape(tf.trace(volT @ Q), [self.size, 1]) + 
            self.f( t * self.h , X, pi)
            )

        ret = tf.gradients(
            H,
            X,
            stop_gradients=[pi, Z, Q],
        )[0]
        return ret

    def f(self, t, X, pi):
        raise NotImplementedError

    def V1_init(self, name):
        var = tf.Variable(
            tf.random_uniform([1, 1], minval=-0.1, maxval=0.1, dtype=tf.float64),
            name=f"V1_0_{name}",
        )
        return var

    def V1_update(self, t, name):
        V = self.processes['V1'][name]
        X = self.processes['X'][name]
        pi = self.processes['primal_control'][name]
        Z = self.processes['Z1'][name]
        drift = self.config.alpha * V[t]  - self.f(t * self.h, X[t], pi[t])
        vol = tf.expand_dims(Z[t], 1) @ self.vol(t * self.h, X[t], pi[t])
        return ( V[t] +
            drift * self.h
            + b_mult(vol, self.dW[:, :, t]) * self.sqrth
        )

    def Gamma1_init(self, name):
        var  = tf.Variable(
            tf.random_uniform(
                [1, self.d, self.d], minval=0.2, maxval=0.3, dtype=tf.float64
            ),
            name=f"Gamma1_0_{name}",
        )
        return  var

    def Gamma1_update(self, t, name):
        if t >= self.N - 1:
            return None
        X = tf.stop_gradient(self.processes['X'][name][t+1])
        current_net, self._extra_train_ops_bsde = net(
            X,
            f"Gamma1_{t+1}_{name}",
            self._extra_train_ops_bsde,
            self.n_neuron_Gamma,
        )
        return tf.reshape(
                current_net, [self.size, self.d, self.d]
            )

    def primal_control_init(self, name):
        var = tf.Variable(
            tf.random_uniform(
                [1, self.m], minval=0.05, maxval=0.051, dtype=tf.float64
            ),
            name=f"primal_control_0_{name}",
        )
        return self.restrict(var)

    def restrict(self, pi):
        raise NotImplementedError

    def primal_control_update(self, t, name):
        if t >= self.N - 1:
            return None
        X = tf.stop_gradient(self.processes['X'][name][t+1])
        current_net, self._extra_train_ops_control = net(
            X,
            f"primal_control_{t+1}_{name}",
            self._extra_train_ops_control,
            self.n_neuron_primal_control
            )
        return self.restrict(current_net)

    def J1_init(self, name):
        return tf.zeros([1, 1], dtype=tf.float64)

    def J1_update(self, t, name):
        X = self.processes['X'][name]
        pi = self.processes['primal_control'][name]
        J = self.processes['J1'][name]
        discount = self.config.alpha
        ret = exp(- discount * self.h * t) * self.h * self.f(t * self.h, X[t], pi[t])
        if t == self.N - 1:
            ret += exp(- discount * self.config.T) * self.g(X[self.N])
        return J[t] + ret

    def g(self, X):
        raise NotImplementedError




    def get_vars_primal(self):
        ret = {}
        ret['y'] = None
        ret['bsde'] = {}
        ret['bsde'][0] = tf.trainable_variables("V1_0_primal") + tf.trainable_variables("Z1_0_primal") + tf.trainable_variables("Gamma1_0_primal")
        ret['control'] = {}
        ret['control'][0] = tf.trainable_variables("primal_control_0_primal")
        for t in range(1,self.N):
            ret['control'][t] = tf.trainable_variables(
                f"forward/primal_control_{t}_primal"
            )
            ret['bsde'][t] = tf.trainable_variables(
                f"forward/Gamma1_{t}_primal"
            )
            
        return ret


    def get_losses_primal(self):
        V = self.processes['V1']['primal']
        X = self.processes['X']['primal']
        Z = self.processes['Z1']['primal']
        beta1 = 1.0
        beta2 = 5.0
        # primal_control = self.processes['primal_control']['primal']

        delta1 = V[self.N] - self.g(X[self.N])
        delta2 = Z[self.N] - self.grad_g(X[self.N])
        bsde = [
            tf.reduce_mean(beta1 * tf.square(delta1) + beta2 * b_dot(delta2,delta2))
            # tf.reduce_mean(beta2 * b_dot(delta2,delta2))
#            + tf.reduce_sum([self.F(t) for t in range(self.N)])
            for _ in range(self.N)]
        y = None
        control = [
            tf.reduce_mean(self.F(t))
 #           + tf.reduce_mean(tf.square(delta1) + 0.5 * b_dot(delta2,delta2))
             # tf.reduce_mean(tf.reduce_sum(tf.square(primal_control[t] - self.config.solution.primal_control(t * self.h, None)), 1, keepdims = True))
            for t in range(self.N)
        ]
        return bsde, control, y

    def grad_g(self, X):
        return tf.gradients(self.g(X), X)[0]

    def F(self, t):
        X     = tf.stop_gradient(self.processes["X"]             ['primal'][t])
        Z     = tf.stop_gradient(self.processes["Z1"]            ['primal'][t])
        Gamma = tf.stop_gradient(self.processes["Gamma1"]        ['primal'][t])
        pi    =                  self.processes["primal_control"]['primal'][t]

        
        volT = tf.linalg.transpose(self.vol(t * self.h, X, pi))
        
        drift_term = b_dot(self.drift(t * self.h, X, pi), Z)
        vol_term = tf.reshape(
            tf.trace(self.vol(t * self.h, X, pi) @ volT @ Gamma),
            [self.size, 1],
        )
        f_term = self.f(t * self.h, X, pi)
        return -1 * (drift_term + 0.5 * vol_term + f_term)

    def load_bruteprimal(self):
        self.process_keys['bruteprimal'] = ['X', 'primal_control', 'J1']
        self.n_neuron_primal_control = (
                [self.d] + [self.d + self.config.extra] * self.config.layers + [self.m]
        )

    def get_vars_bruteprimal(self):
        ret = {}
        ret['y'] = None
        ret['bsde'] = {}
        ret['bsde'][0] = None
        ret['control'] = {}
        ret['control'][0] = tf.trainable_variables("primal_control_0_bruteprimal")
        for t in range(1,self.N):
            ret['control'][t] = tf.trainable_variables(
                f"forward/primal_control_{t}_bruteprimal"
            )
            ret['bsde'][t] = None
        return ret

    def get_losses_bruteprimal(self):
        bsde = [None for _ in range(self.N)]
        y = None
        control = [
            -tf.reduce_mean(self.processes['J1']['bruteprimal'][self.N])
            for t in range(self.N)
        ]
        return bsde, control, y


    def load_hybrid(self):
        self.process_keys['hybrid'] = ['X', 'primal_control', 'V', 'J1']
        self.n_neuron_primal_control = (
                [self.d] + [self.d + self.config.extra] * self.config.layers + [self.m]
        )
        self.n_neuron_value = (
                [self.d] + [self.d + self.config.extra] * self.config.layers + [1]
        )

    def V_terminal(self, name):
        return self.g(self.processes['X'][name][self.N])

    def V_update(self, t, name):
        X = self.processes['X'][name]
        #V = self.processes['V'][name]
        if t > 0:
            current_net, self._extra_train_ops_control = net(
                X[t],
                f"value_{t}_{name}",
                self._extra_train_ops_bsde,
                self.n_neuron_value
                )
            ret = current_net + X[t]
        else:
            var = tf.Variable(
                tf.random_uniform([1, 1], minval=0.9, maxval=1.1, dtype=tf.float64),
                name=f"value_0_{name}",
            )

            try:
                self.V1_0[name] = var
            except AttributeError:
                self.V1_0 = {}
                self.V1_0[name] = var
            ret =  self.allones @ self.V1_0[name]


        return ret


    def get_vars_hybrid(self):
        ret = {}
        ret['y'] = None
        ret['bsde'] = {}
        ret['control'] = {}

        for t in range(self.N):
            ret['control'][t] = tf.trainable_variables(
                f"forward/primal_control_{t}_hybrid"
            )
            ret['bsde'][t] = tf.trainable_variables(
                f"forward/value_{t}_hybrid"
            )
        ret['control'][0] = tf.trainable_variables("primal_control_0_hybrid")
        return ret

    def get_losses_hybrid(self):
        X = self.processes['X']['hybrid']
        primal_control = self.processes['primal_control']['hybrid']
        V = self.processes['V']['hybrid']

        control = [
            -1 * (t + 1) / (self.N + 1) * tf.reduce_mean(self.h  * self.f(t*self.h, X[t], primal_control[t]) + V[t+1]) #+ (V[t + 2] if t < self.N - 1 else 0))

            # -1 * tf.reduce_mean(self.h  * self.f(t*self.h, X[t], primal_control[t]) + V[self.N]) #+ (V[t + 2] if t < self.N - 1 else 0))
            # tf.reduce_mean(tf.reduce_sum(tf.square(primal_control[t] - self.config.solution.primal_control(t * self.h, None)), 1, keepdims = True))
            for t in range(self.N)
            ]

        y = None
        bsde = [
            # tf.reduce_mean(tf.square( self.config.solution.u(t * self.h, X[t]) - V[t]))
            (t + 1) / (self.N + 1) * tf.reduce_mean(tf.square(self.h * self.f(t*self.h, X[t], primal_control[t]) + V[t+1] - V[t]))
            for t in range(self.N)
        ]
        return bsde, control, y


    def value_op(self, name, monte):
        if name in ['primal', 'hybrid'] and not monte:
            return [self.V1_0[name][0][0]]
        else:
            return [tf.reduce_mean(self.processes['J1'][name][self.N])]

    def sample(self, num_sample, _k=None, _N=None, rand=True):
        k = _k if _k else self.k
        N = _N if _N else self.N
        random_state = None if rand else 1234
        return {self.dW: normal.rvs(size=[num_sample, k, N], random_state = random_state).reshape((num_sample, k, N))}

    def get_ones(self, k, l=None):
        if l:
            return tf.ones(
                shape=tf.stack([self.size, k, l]), dtype=tf.float64
            )
        else:
            return tf.ones(
                shape=tf.stack([self.size, k]), dtype=tf.float64
            )







class UtilityMethod(Method):
    def __init__(self, config, names, graph, monte):
        super().__init__(config, names, graph, monte)
        self.m_dual = self.config.m_dual
        self.modC = True  #  c -> c * X
        self.modGamma = True # gam -> gam * Y
        self.modPi = False # X * pi - > pi
        ones_k = self.get_ones(self.k)
        ones_kk = self.get_ones(self.k, self.k)
        self.theta = lambda s : ones_k * self.config.theta(s)
        self.theta_mod = self.config.theta_mod
        self.b = lambda s : ones_k * (self.config.mu(s) - self.config.r(s))
        self.sigma = lambda s : ones_kk * self.config.sigma(s)
        self.sigma_inv = lambda s : ones_kk * np.linalg.inv(self.config.sigma(s))
        self.sigma_trans = lambda s : ones_kk * self.config.sigma(s).T
        self.sigma_inv_trans = lambda s : ones_kk * np.linalg.inv(self.config.sigma(s).T)

    def fixProcesses(self):

        if 'primal_control' in self.processes.keys():
            for name in self.processes['primal_control'].keys():
                for t in range(self.N - 1):
                    pi = self.processes['primal_control'][name][t][:,:self.k]
                    if self.config.c:
                        c = self.processes['primal_control'][name][t][:,self.k:]
                    if self.modPi:
                        pi *= self.processes['X'][name][t]

                    if self.modC and self.config.c:
                        c *= 1 / self.processes['X'][name][t]
                    if self.config.c:
                        self.processes['primal_control'][name][t] = tf.concat([pi, c], 1)
                    else:
                        self.processes['primal_control'][name][t] = pi

        if self.config.gamma:
            if 'dual_control' in self.processes.keys():
                for name in self.processes['dual_control'].keys():
                    for t in range(self.N - 1):
                        v = self.processes['dual_control'][name][t][:,:self.k]
                        gam = self.processes['dual_control'][name][t][:,self.k:]
                        if self.modGamma:
                            try:
                                gam *= 1 / self.processes['Y'][name][t]
                            except KeyError:
                                gam *= 1 / self.processes['Z1'][name][t]

                        self.processes['dual_control'][name][t] = tf.concat([v, gam], 1)

        if 'smp' in self.names:
            for t in range(self.N):
                self.processes['Q']['smp'][t] = b_mult(self.sigma_trans(t * self.h), self.processes['Q']['smp'][t])
        super().fixProcesses()


    def restrict(self, a):
        pi = a[:,:self.k]
        if self.config.space == "whole":
            pi = pi + 1
        elif self.config.space == "cone":
            pi = tf.maximum(pi, 0)
        elif self.config.space == "zero":#
            pi = pi * 0.0

        if self.config.c:
            c = a[:,self.k:]
            return tf.concat([pi,tf.square(c)],1)
        else:
            return pi

    def drift(self, t, X, a):
        pi = a[:,:self.k]
        if self.modPi:
            pi *= 1 / X
        ret = b_dot(X, b_dot(pi, self.b(t)) + self.config.r(t)) + self.config.a

        if self.config.c:
            c = a[:,self.k:]
            if self.modC:
                c *= X
            ret += -c

        return ret


    def vol(self, t, X, a):  # d *
        pi = a[:,:self.k]
        if  self.modPi:
            pi*= 1 / X
        right = tf.expand_dims(pi, 1) @ self.sigma(t)
        return tf.expand_dims(X, -1) @ right


    def f(self, t, X, a):
        ret = 0
        if self.config.gamma:
            ret += self.config.eta * self.u(t,X)

        if self.config.c :
            c = a[:,self.k:]
            if self.modC:
                c *= X
            ret += self.U(c)

        if self.config.space == 'ball':
            pi = a[:,:self.k]
            norm = b_dot(pi, pi)
            ret += -1000 * tf.square(tf.maximum(tf.zeros_like(norm), norm - 1))

        return ret



    def dual_restrict(self, a):
        v = a[:,:self.k]

        if self.config.space == "whole":
            v = v * 0.0
        elif self.config.space == "cone":
            v = tf.square(v + 0.5)

        if self.config.gamma:
            gam = a[:,self.k:]
            return tf.concat([v, tf.maximum(gam+1,tf.zeros_like(gam))],1)
        else:
            return v

    def drift_dual(self, t, Y, a):
        v = a[:,:self.k]
        ret = Y * (self.config.alpha - self.config.r(t) - self.delta(v))
        if self.config.gamma:
            gam = a[:,self.k:]
            if self.modGamma:
                gam *= Y
            ret += - self.config.eta * gam
        return ret

    def vol_dual(self, t, Y, a):  # d * k
        v = a[:,:self.k] # 1 x k
        sig_v = b_mult(self.sigma_inv(t), v)

        return -1 * tf.expand_dims(Y, -1) @ tf.expand_dims(self.theta(t) + sig_v,1)



    def f_dual(self, t, Y, a):
        ret = self.config.a * Y
        if self.config.c:
            ret += self.U_dual(Y)
        if self.config.gamma:
            gam = a[:,self.k:]
            if self.modGamma:
                gam *= Y
            ret += self.config.eta * self.v(t, gam)

        return ret

    def dual_control_update(self, t, name):
        if t >= self.N - 1:
            return None
        if name == 'primal':
            Gamma = self.processes['Gamma1'][name][t+1]
            X = self.processes['X'][name][t+1]
            pi = self.processes['primal_control'][name][t+1][:, :self.k]
            Z = self.processes['Z1'][name][t+1]
            Q = tf.squeeze(Gamma @ self.vol((t+1) * self.h, X, pi), 1)
            sig = self.sigma((t+1)*self.h)
            theta = self.theta((t+1)*self.h)
            v = -1 * b_mult(sig, Q/Z + theta)
            if self.config.gamma:
                gam = Z[t+1]
                if self.modGamma:
                    gam *= 1 / X[t+1]
                return tf.concat([v, gam], 1)
            else:
                return v
        else:
            Y = self.processes['Y'][name][t+1]
            current_net, self._extra_train_ops_control = net(
                Y,
                f"dual_control_{t + 1}_{name}",
                self._extra_train_ops_control,
                self.n_neuron_dual_control,
            )

            return self.dual_restrict(current_net)


    def primal_control_update(self, t, name):
        if t >= self.N - 1:
            return None
        if name in ['dual', 'smp']:
            Y = self.processes['Y'][name][t+1]
            Z = self.processes['Z2'][name][t+1]
            v = self.processes['dual_control'][name][t+1][:, :self.k]

            if name == 'dual':
                Gamma = self.processes['Gamma2'][name][t+1] #B x d x d
                Q = Gamma[:,:1,:1] @ self.vol_dual((t+1) * self.h, Y, v)[:,:1,:self.m] # b x d x k
                sig_inv = self.sigma_inv((t+1) * self.h) # b x m x m
                Q = b_mult(sig_inv, Q[:,0,:self.m])
            else:
                Q = self.processes['Q'][name][t+1]
            pi =  Q / Z
            if self.modPi:
                pi * -1 * Z[t+1]

            if self.config.c:
                c = self.I(Y[t+1])
                if self.modC:
                    c*= -1 /  Z[t+1]
                return tf.concat([pi,c], 1)
            else:
                return pi
        else:
            return super().primal_control_update(t, name)



    def load_dual(self):
        self.process_keys['dual'] = ['Y',  'Z2', 'V2', 'Gamma2', 'dual_control', 'J2']
        self.process_keys['dual'] += ['X', 'Z1', 'V1', 'Gamma1', 'primal_control', 'Q']
        
        self.n_neuron_Gamma = (
            [self.d] + [self.d + self.config.extra] * self.config.layers + [self.d ** 2]
        )
        self.n_neuron_dual_control = (
                [self.d] + [self.d + self.config.extra] * self.config.layers + [self.config.m_dual]
                )

    def load_smp(self):
        self.process_keys['smp'] = ['Y',  'dual_control', 'Z2','Q','J2',  'X', 'Z1', 'primal_control','J1']
        self.n_neuron_dual_control = (
                [self.d] + [self.d + self.config.extra] * self.config.layers + [self.config.m_dual]
        )
        self.n_neuron_Q = (
                [self.d] + [self.d + self.config.extra] * self.config.layers + [self.m]
            )



    def X_init(self, name):
        if name in [ 'dual', 'smp', 'bsde']:
            var = -1 * self.Z2_0[name][0,:1]

            if self.d > 1:
                var = tf.concat([var, self.config.Xinit[1:]], 0)
            try:
                self.X_0[name] = var
            except AttributeError:
                self.X_0 = {}
                self.X_0[name] = var
            return tf.ones(shape = tf.stack([self.size, self.d]), dtype=tf.float64) * self.X_0[name]
        else:
             return super().X_init(name)


    def X_update(self, t, name):
         if name  in ['dual', 'bsde']:
             return -1 * self.processes['Z2'][name][t+1]
         elif name == 'smp':
             if self.d > 1:
                 V = self.processes['Y'][name][t+1][:,1:]
                 Z = self.processes['Z2'][name][t+1]
                 return tf.concat([-Z, V], 1)
             else:
                 return -1 * self.processes['Z2'][name][t+1]
         else:
             X = self.processes['X'][name]
             pi = self.processes['primal_control'][name]
             ret =  tf.maximum(( X[t] +
                 self.drift(t * self.h, X[t], pi[t]) * self.h
                 + b_mult(self.vol(t * self.h, X[t], pi[t]), self.dW[:, :, t]) * self.sqrth
                 ), 1e-4)
             return ret



    def Z1_init(self, name):
        if name in ['dual', 'smp', 'brutedual', 'bsde']:
            return self.Z1_update(-1, name)[:1]
        else:
            return super().Z1_init( name)

    def Z1_update(self, t, name):
        if name in ['dual', 'smp', 'brutedual', 'bsde']:
            return self.processes['Y'][name][t+1]
        else:
            return super().Z1_update(t, name)

    def grad_H(self, t, X, a, Z, Q):
        pi = a[:,:self.k]
        volT =  tf.linalg.transpose(tf.expand_dims(pi, 1) @ self.sigma(t))
        drift = b_dot(pi, self.b(t)) + self.config.r(t)
        dH = b_dot(drift, Z) + tf.reshape(tf.trace(volT @ Q), [self.size, 1])
        return dH

    def V1_init(self, name):
        if name in ['dual', 'bsde']:
            return self.V1_update(-1, name)[:1]
        else:
            return super().V1_init( name)

    def V1_update(self, t, name):
        if name in ['dual', 'bsde']:
            return self.processes['V2'][name][t+1] - self.processes['Z2'][name][t+1][:,:1] * self.processes['Y'][name][t+1][:,:1]
        else:
            return super().V1_update(t, name)

    def Gamma1_init(self, name):
        if name in ['dual']:
            return self.Gamma1_update(-1, name)[:1]
        else:
            return super().Gamma1_init( name)

    def Gamma1_update(self, t, name):
        if t >= self.N - 1:
            return None
        if name in ['dual']:
            return -1 / self.processes['Gamma2'][name][t+1]
        else:
            return super().Gamma1_update(t, name)

    def primal_control_init(self, name):
        if name in ['dual', 'smp']:
            return self.primal_control_update(-1, name)[:1]
        else:
            return super().primal_control_init( name)




    def J1_update(self, t, name):
        X = self.processes['X'][name]
        pi = self.processes['primal_control'][name]
        J = self.processes['J1'][name]
        discount = self.config.alpha
        ret = exp(- discount * self.h * t) * self.h * self.f(t * self.h, X[t], pi[t])
        if t == self.N - 1:
            ret += exp(- discount * self.config.T) * self.g(X[self.N])
        return J[t] + ret

    def Y_init(self, name):
        if name == 'primal':
            var = self.Z1_0[name][:1,:1]
        else:
            var = tf.Variable(
                        tf.random_uniform([1, 1], minval= 0.2, maxval = 0.3 , dtype=tf.float64), #
                        name=f"Y_0_{name}",
                    )
            if 1 / self.config.Xinit[0] <= 2e-2:
                raise ValueError("Change Y_init")
            var = tf.maximum(var, 1e-4) #* 0 + self.config.solution.Du(0, self.config.Xinit[0])

        if self.d > 1:
            var = tf.concat([var, tf.ones([1,self.config.d - 1], dtype = tf.float64) * self.config.Xinit[1:]], 1)
        return var

    def Y_update(self, t, name):
        if name == 'primal':
            return self.processes['Z1'][name][t+1]
        else:
            Y = self.processes['Y'][name]
            gam = self.processes['dual_control'][name]
            return ( Y[t]
                + self.drift_dual(t * self.h, Y[t], gam[t]) * self.h
                + b_mult(self.vol_dual(t * self.h, Y[t], gam[t]), self.dW[:, :, t]) * self.sqrth
            )

    def delta(self, v):
        if self.config.space in ["whole", "cone", "zero"]:
            return tf.zeros([self.size,1], dtype = tf.float64)
        elif self.config.space == "ball":
            return tf.norm(v, axis=1, keepdims=True)
        else:
            raise ValueError("cannot understand config.space")

    def V2_init(self, name):
        var = tf.Variable(
            tf.random_uniform([1, 1], minval=-0.1, maxval=0.1, dtype=tf.float64),
            name=f"V2_0_{name}",
        )
        return var

    def V2_update(self, t, name):
        Y = self.processes['Y'][name]
        gam = self.processes['dual_control'][name]
        Z = self.processes['Z2'][name]
        V = self.processes['V2'][name]
        drift = self.config.alpha * V[t] - self.f_dual(t * self.h, Y[t], gam[t])
        vol = tf.expand_dims(Z[t], 1) @ self.vol_dual(t * self.h, Y[t], gam[t])
        return (V[t]
            + drift * self.h
            + b_mult(vol, self.dW[:, :, t]) * self.sqrth
        )


    def Z2_init(self, name):

        var =  - tf.Variable(self.config.Xinit[np.newaxis, :1], trainable=False)
        if name == 'dual' and self.d > 1:
            temp = tf.Variable(
                tf.random_uniform([1, self.d - 1], minval=0.9, maxval=1.1, dtype=tf.float64),
                name=f"Z2_0_{name}",
            )
            var = tf.concat([var, temp], 1)
        return var

    def Z2_update(self, t, name):
        Y = self.processes['Y'][name]
        gam = self.processes['dual_control'][name]
        Z = self.processes['Z2'][name]
        if name == 'dual':
            Gamma = self.processes['Gamma2'][name]
            # print(Gamma[t], self.vol_dual(t * self.h, Y[t], gam[t]) )
            Q = (Gamma[t] @ self.vol_dual(t * self.h, Y[t], gam[t]))
        elif name == 'smp':
            Q =  tf.expand_dims(self.processes['Q'][name][t],1) @ self.sigma(t * self.h)


        drift = self.config.alpha * Z[t] - self.grad_H_dual(t, Y[t], gam[t], Z[t], tf.linalg.transpose(Q), name)
        return (Z[t] +
            drift * self.h
            + b_mult(Q, self.dW[:, :(self.m if name == 'smp' else self.k) , t]) * self.sqrth
        )

    def grad_H_dual(self, t, Y, a, Z, Q, name):
        v = a[:,:self.k]
        drift = (self.config.alpha - self.config.r(t) - self.delta(v))
        sig_v = self.sigma_inv(t) @ tf.expand_dims(v, -1)
        volT = -1 * tf.linalg.transpose(
            (tf.expand_dims(self.theta(t),-1) + sig_v)
            )


        dH = drift * Z + tf.reshape(tf.trace( volT @ Q), [self.size, 1])
        return dH






    def Gamma2_init(self, name):
        var = tf.Variable(
            tf.random_uniform(
                [1, self.d, self.d], minval= 0.01, maxval=0.1, dtype=tf.float64
            ),
            name=f"Gamma2_0_{name}",
        )
        return tf.exp(var)

    def Gamma2_update(self, t, name):
        if t >= self.N - 1:
            return None
        Y = self.processes['Y'][name]

        current_net, self._extra_train_ops_bsde = net(
            Y[t+1],
            f"Gamma2_{t + 1}_{name}",
            self._extra_train_ops_bsde,
            self.n_neuron_Gamma,
        )
        return tf.reshape(
                tf.exp(current_net), [self.size, self.d, self.d]
            )

    def dual_control_init(self, name):
        if name == 'primal':
            return self.dual_control_update(-1, name)
        var = tf.Variable(
            tf.random_uniform(
                [1, self.m_dual], minval=0.05, maxval=0.051, dtype=tf.float64
            ),
            name=f"dual_control_0_{name}",
        )
        return self.dual_restrict(var)




    def J2_init(self, name):
        return tf.zeros(shape=tf.stack([self.size, 1]), dtype=tf.float64)

    def J2_update(self,t, name):
        Y = self.processes['Y'][name]
        gam = self.processes['dual_control'][name]
        J = self.processes['J2'][name]
        discount = self.config.alpha
        ret = exp(- discount * self.h * t) * self.h * self.f_dual(t * self.h, Y[t], gam[t]) * (0.5 if t in [0, self.N - 1] else 1)
        if t == self.N - 1:
            _Y = Y[self.N]
            if name == 'primal':
                _Y = tf.maximum(_Y, 1e-2)
            ret += exp(- discount * self.config.T) * self.g_dual(_Y)
            ret += self.Y_0[name][0] * self.config.Xinit[0]
        return J[t] + ret

    def g_dual(self, Y):
        raise NotImplementedError






    def get_vars_dual(self):
        ret = {}
        ret['y'] =  tf.trainable_variables("Y_0_dual")
        ret['bsde'] = {}
        ret['bsde'][0] = tf.trainable_variables("Z2_0_dual") + tf.trainable_variables("V2_0_dual") + tf.trainable_variables("Gamma2_0_dual")

        ret['control'] = {}
        ret['control'][0] = tf.trainable_variables("dual_control_0_dual")
        for t in range(1,self.N):
            ret['control'][t] = tf.trainable_variables(
                "forward/dual_control_" + str(t) + "_dual"
            )
            ret['bsde'][t] = tf.trainable_variables(
                "forward/Gamma2_" + str(t) + "_dual"
            )
        return ret

    def get_losses_dual(self):
        processes = self.processes
        delta1 = processes["V2"]['dual'][self.N] - self.g_dual(processes["Y"]['dual'][self.N])
        delta2 = processes["Z2"]['dual'][self.N] - self.grad_g_dual(processes["Y"]['dual'][self.N])
        loss_bsde = [tf.reduce_mean(tf.square(delta1) + 0.5 * b_dot(delta2, delta2)) for _ in range(self.N)]
        loss_y  = tf.reduce_mean(processes["J2"]['dual'][self.N])
        loss_control= [
            # tf.reduce_mean(processes["J2"]['dual'][self.N])
            tf.reduce_mean(self.F_dual(t))
            for t in range(self.N)
        ]

        return loss_bsde, loss_control, loss_y

    def grad_g_dual(self, Y):
        return tf.gradients(self.g_dual(Y), Y)[0]

    def F_dual(self, t):
        Y= self.processes["Y"]['dual']
        gam = self.processes["dual_control"]['dual']
        Z = self.processes["Z2"]['dual']
        Gamma = self.processes["Gamma2"]['dual']

        volT = tf.linalg.transpose(self.vol_dual(t * self.h, Y[t], gam[t]))
        drift_term = b_dot(self.drift_dual(t * self.h, Y[t], gam[t]), Z[t])
        vol_term = tf.reshape(
            tf.trace(self.vol_dual(t * self.h, Y[t], gam[t]) @ volT @ Gamma[t]),
            [self.size, 1],
        )
        f_term = self.f_dual(t * self.h, Y[t], gam[t])
        return drift_term + 0.5 * vol_term + f_term


    def load_brutedual(self):
        self.process_keys['brutedual'] = ['Y', 'dual_control', 'J2']
        if self.graph:
            self.process_keys['brutedual'] += ['Z1']
        self.n_neuron_dual_control = (
                [self.d] + [self.d + self.config.extra] * self.config.layers + [self.config.m_dual]
                )

    def get_vars_brutedual(self):
        ret = {}
        ret['y'] = tf.trainable_variables("Y_0_brutedual")
        ret['bsde'] = {}
        ret['bsde'][0] = None

        ret['control'] = {}
        ret['control'][0] = tf.trainable_variables("dual_control_0_brutedual")
        for t in range(1,self.N):
            ret['control'][t] = tf.trainable_variables(
                f"forward/dual_control_{t}_brutedual"
            )
            ret['bsde'][t] = None
        return ret

    def get_losses_brutedual(self):
        processes = self.processes
        loss_bsde = [None for _ in range(self.N)]
        loss_y  = tf.reduce_mean(processes["J2"]['brutedual'][self.N])
        loss_control= [
            tf.reduce_mean(processes["J2"]['brutedual'][self.N])
            for t in range(self.N)
        ]

        return loss_bsde, loss_control, loss_y




    def Q_init(self, name):
        if name in ['dual', 'primal']:
            return self.Q_update(-1, name)[:1]
        else:
            var = tf.Variable(
                tf.random_uniform(
                    [1, self.m], minval=0.05, maxval=0.051, dtype=tf.float64
                ),
                name=f"Q_0_{name}",
            )
            return self.getQ(0, self.allones @ var)[:1]

    def Q_update(self, t, name):
        if t >= self.N - 1:
            return None
        if name == 'dual':
            gamma = self.processes['Gamma2']['dual'][t+1] # B x 1 x 1
            Y = self.processes['Y']['dual']
            v = self.processes['dual_control']['dual']
            vol = self.vol_dual(t+1, Y[t+1], v[t+1]) # B x 1 x m

            return (gamma @ vol)[:,0,:]
        elif name == 'primal':
            pi = self.processes['primal_control'][name]
            X = self.processes['X'][name]
            return -1 * self.vol((t+1) * self.h, X[t+1], pi[t+1])[:,0,:]
        else:
            Y = self.processes['Y'][name]

            current_net, self._extra_train_ops_bsde = net(
                Y[t+1],
                "Q_" + str(t + 1) + "_"+ name,
                self._extra_train_ops_bsde,
                self.n_neuron_Q,
            )
            return self.getQ(t + 1, tf.reshape(
                    current_net, [self.size, self.n_neuron_Q[-1]]
                ))



    def get_vars_smp(self):
        ret = {}
        ret['y'] = tf.trainable_variables("Y_0_smp")
        ret['bsde'] = {}
        ret['bsde'][0] = tf.trainable_variables("Q_0_smp")
        ret['control'] = {}
        ret['control'][0] = tf.trainable_variables("dual_control_0_smp")
        for t in range(1,self.N):
            ret['control'][t] = tf.trainable_variables(
                f"forward/dual_control_{t}_smp"
            )
            ret['bsde'][t] = tf.trainable_variables(
                f"forward/Q_{t}_smp"
            )
        return ret

    def get_losses_smp(self):
        P = self.processes['Z2']['smp']
        #Q = self.processes['Q']['smp']
        #v = self.processes['dual_control']['smp']
        Y = self.processes['Y']['smp']
        delta = P[self.N] - self.grad_g_dual(Y[self.N])[:,:1]
        loss_bsde = [tf.reduce_mean(tf.square(delta)) for t in range(self.N)]
        loss_y =  tf.reduce_mean(self.processes["J2"]['smp'][self.N])


        loss_control = [
            # tf.reduce_mean(self.processes["J2"]['smp'][self.N])
            self.smp_loss(t)
            for t in range(self.N)
        ]

        return loss_bsde, loss_control, loss_y

    def getQ(self, t, N):
        P = self.processes["Z2"]['smp']
        # Y = self.processes['Y']['smp']
        # v = self.processes['dual_control']['smp']
        # vol = self.vol_dual(t * self.h, Y[t], v[t]) # 1 x m
        # N = b_mult(vol, N)
        N = self.restrict(N)
        return N  * P[t]


    def smp_loss(self, t):
        P = self.processes['Z2']['smp']
        Q = self.processes['Q']['smp']
        v = self.processes['dual_control']['smp']
        #Y = self.processes['Y']['smp']


        ret =  tf.reduce_mean([tf.square(P[s] * self.delta(v[s]) +  b_dot(Q[s], v[s])) for s in range(t, min(t+1, self.N))])
        # if self.config.space == 'cone':
        #     ret += 100 * tf.reduce_sum([self.constrain_loss(s) for s in range(t, min(t+2, self.N))])

        return ret
        # return tf.reduce_mean(self.processes["J2"]['smp'][self.N])


    def constrain_loss(self, s):
        pi = self.processes['primal_control']['smp']
        if self.config.space == 'cone':
            return b_dot(tf.minimum(tf.zeros_like(pi[s]),pi[s]),tf.minimum(tf.zeros_like(pi[s]),pi[s]))
        else:
            return tf.zeros()


    def load_pde(self):
        assert self.config.space in ['whole', 'cone', 'zero'], 'cant use pde method for this space'
        self.process_keys['pde'] = ['Xp', 'Zp', 'Vp', 'Gp', 'Ap']
        self.n_neuron = (
                [1] + [self.d + self.config.extra] * self.config.layers + [1]
                )

    def Xp_init(self, name):
        return super().X_init(name)


    def Xp_update(self, t, name):
        X = self.processes['Xp'][name][t]
        dw =  self.dW[:, :1, t] * self.sqrth
        return X + dw


    def Zp_init(self, name):
        return super().Z1_init( name)

    def Zp_update(self, t, name):
        A = self.processes['Ap']['pde'][t]
        Gamma = self.processes['Gp']['pde'][t]
        Z = self.processes['Zp']['pde'][t]

        return Z + A * self.h + b_dot(Gamma, self.dW[:, :1, t]) * self.sqrth

    def Vp_init(self, name):
        return super().V1_init( name)

    def Vp_update(self, t, name):
        X = self.processes['Xp'][name][t]
        Gamma = self.processes['Gp']['pde'][t]

        Z = self.processes['Zp']['pde'][t]
        V = self.processes['Vp']['pde'][t]

        drift = np.dot(self.config.solution.theta_hat(t * self.h), self.config.theta(self.h * t)) * tf.square(Z) / (2 * Gamma) - self.config.r(self.h * t) * X * Z# + Gamma / 2

        return V + drift * self.h + b_dot(Z, self.dW[:, :1, t]) * self.sqrth



    def Ap_init(self, name):
        var  = tf.Variable(
            tf.random_uniform(
                [1, 1], minval=-0.1, maxval=-0.01, dtype=tf.float64
            ),
            name=f"Ap_0_{name}",
        )
        return var

    def Ap_update(self, t, name):
        if t >= self.N - 1:
            return None
        X = self.processes['Xp'][name]
        current_net, self._extra_train_ops_bsde = net(
            X[t+1],
            f"Ap_{t + 1}_{name}",
            self._extra_train_ops_bsde,
            self.n_neuron,
        )
        return current_net / 10

    def Gp_init(self, name):
        var  = tf.Variable(
            tf.random_uniform(
                [1, 1], minval=-0.2, maxval=0.2, dtype=tf.float64
            ),
            name=f"Gp_0_{name}",
        )

        return -1 * tf.exp(var)

    def Gp_update(self, t, name):
        if t >= self.N - 1:
            return None
        X = self.processes['Xp'][name]
        current_net, self._extra_train_ops_bsde = net(
            X[t+1],
            f"Gp_{t + 1}_{name}",
            self._extra_train_ops_bsde,
            self.n_neuron,
        )
        return -1 * tf.exp(current_net / 10)



    def get_vars_pde(self):
        ret = {}
        ret['y'] = None
        ret['bsde'] = {}
        ret['bsde'][0] = (
            tf.trainable_variables("V1_0_pde") + tf.trainable_variables("Z1_0_pde")
            + tf.trainable_variables("Ap_0_pde") + tf.trainable_variables("Gp_0_pde")
            )
        ret['control'] = {}
        ret['control'][0] = None
        for t in range(1,self.N):
            ret['control'][t] = None
            ret['bsde'][t] = tf.trainable_variables(
                f"forward/Ap_{t}_pde"
            ) + tf.trainable_variables(
                f"forward/Gp_{t}_pde"
            )

        return ret


    def get_losses_pde(self):
        V = self.processes['Vp']['pde']
        X = self.processes['Xp']['pde']
        #Z = self.processes['Zp']['pde']

        delta1 = V[self.N] - self.g(X[self.N])
        #delta2 = Z[self.N] - self.grad_g(X[self.N])
        bsde = [tf.reduce_mean(tf.square(delta1)) for _ in range(self.N)]
        y = None
        control = [
            None
            for _ in range(self.N)
        ]
        return bsde, control, y

    def value_op(self, name, monte):
        if name == 'primal':
            if 'J2' in self.processes.keys() and 'primal' in self.processes['J2'].keys():
                if monte:
                    return [tf.reduce_mean(self.processes['J1']['primal'][self.N]), tf.reduce_mean(self.processes['J2']['primal'][self.N])]
                else:
                    return [self.V1_0['primal'][0][0], tf.reduce_mean(self.processes['J2']['primal'][self.N])]
            else:
                if monte:
                    return [tf.reduce_mean(self.processes['J1']['primal'][self.N])]
                else:
                    return [self.V1_0['primal'][0][0]]
        if name == 'dual':
            if 'J1' in self.processes.keys() and 'dual' in self.processes['J1'].keys():
                if monte:
                    return [tf.reduce_mean(self.processes['J1']['dual'][self.N]), tf.reduce_mean(self.processes['J2']['dual'][self.N])]
                else:
                    return [tf.reduce_mean(self.processes['J1']['dual'][self.N]), self.V2_0['dual'][0][0] + self.Y_0['dual'][0][0] * self.config.Xinit[0]]
            else:
                if monte:
                    return [tf.reduce_mean(self.processes['J2']['dual'][self.N])]
                else:
                    return [self.V2_0['dual'][0][0] + self.Y_0['dual'][0][0] * self.config.Xinit[0]]
        elif name == 'smp':
            return [tf.reduce_mean(self.processes['J1']['smp'][self.N]), tf.reduce_mean(self.processes['J2']['smp'][self.N])]
        elif name == 'brutedual':
            return [tf.reduce_mean(self.processes['J2']['brutedual'][self.N])]
        elif name == 'pde':
            return [self.Vp_0['pde'][0][0]]
        else:
            return super().value_op(name, monte)



















class PowMethod(UtilityMethod):
    def g(self, X):
        X_safe = tf.where(X > 0, X, tf.ones_like(X) * 1e-5)
        return tf.where(
            X > 0, tf.pow(X_safe, self.config.p) / self.config.p, tf.zeros_like(X)
        )

    def I(self, X):
        X_safe = tf.where(X > 0, X, tf.ones_like(X) * 1e-5)
        return tf.where(
            X > 0, tf.pow(X_safe, 1 / (self.config.p - 1)), tf.zeros_like(X)
        )

    def g_dual(self, Y):
        Y_safe = tf.where(Y > 0, Y, tf.ones_like(Y) * 1e-5)
        return tf.where(
            Y > 0,
            -self.config.q * tf.pow(Y_safe, self.config.q),
            tf.zeros_like(Y),
        )

    def U(self, X):
        X_safe = tf.where(X > 0, X, tf.ones_like(X) * 1e-5)
        return tf.where(
            X > 0, tf.pow(X_safe, self.config.p) / self.config.p, tf.zeros_like(X)
        )

    def u(self, t, X):
        return self.U(X) * self.config.F(t)

    def v(self, t, Y):
        return  self.U_dual(Y) * self.config.G(t)

    def U_dual(self, Y):
        Y_safe = tf.where(Y > 0, Y, tf.ones_like(Y) * 1e-5)
        return tf.where(
            Y > 0,
            -1 * tf.pow(Y_safe, self.config.q) / self.config.q,
            tf.zeros_like(Y),
        )

class LogMethod(UtilityMethod):

    def g(self, X):
        X_safe = tf.where(X > 0, X, tf.ones_like(X) * 1e-5)
        return tf.where(X > 0, tf.log(X_safe), tf.zeros_like(X))

    def grad_g(self, X):
        X_safe = tf.where(X > 0, X, tf.ones_like(X) * 1e-5)
        return tf.where(X > 0, 1 / X_safe, tf.zeros_like(X))

    def I(self, X):
        return self.grad_g(X)

    def g_dual(self, Y):
        Y_safe = tf.where(Y > 0, Y, tf.ones_like(Y) * 1e-5)
        return tf.where(Y > 0, -1 - tf.log(Y_safe), tf.zeros_like(Y))

class HMethod(UtilityMethod):

    def g(self, X):
        X_safe = tf.where(X > 0, X, tf.ones_like(X) * 1e-5)
        H = lambda x: tf.sqrt(2 / (-1 + tf.sqrt(1 + 4 * x)))
        safe_H = (
            (1 / 3) * tf.pow(H(X_safe), -3)
            + tf.pow(H(X_safe), -1)
            + tf.reduce_sum(X_safe * H(X_safe), 1, keepdims=True)
        )
        return tf.where(X > 0, safe_H, tf.zeros_like(X))

    def I(self, X):
        raise NotImplementedError

    def g_dual(self, Y):
        Y_safe = tf.where(Y > 0, Y, tf.ones_like(Y) * 1e-5)
        safe_g = (1 / 3) * tf.pow(Y_safe, -3) + tf.pow(Y_safe, -1)
        return tf.where(Y > 0, safe_g, tf.zeros_like(Y))


class YaariMethod(UtilityMethod):
    def g(self, X):
        return tf.minimum(X, 0.0 * X + self.config.H)

    def grad_g(self, X):
        return tf.where(X > self.config.H, X * 0.0, X * 0.0 + 1.0)

    def I(self, X):
        raise ValueError('Nope')

    def g_dual(self, Y):
        return self.config.H * tf.maximum(0.0 * Y, 1.0 - Y)

    def grad_g_dual(self, Y):
        return self.config.H * tf.where(Y > 1, tf.zeros_like(Y), -1 * tf.ones_like(Y))


class QuadMethod(UtilityMethod):
    def g(self, X):
        return tf.where(
            X < self.config.beta * self.config.H,
            0.5 * self.config.C * X ** 2 - self.config.C * self.config.H * X,
            (self.config.C * self.config.beta * self.config.H ** 2 * (0.5 * self.config.beta - 1)) * tf.ones_like(X)
            )

    def grad_g(self, X):
        return tf.where(
            X < self.config.beta * self.config.H,
            self.config.C * (X - self.config.H),
            tf.zeros_like(X)
            )

    def I(self, X):
        raise ValueError('Nope')

    def g_dual(self, Y):
        return tf.where(
            Y < self.config.C * self.config.H * (self.config.beta - 1),
            self.config.beta * self.config.H * (self.config.C*self.config.H*(self.config.beta / 2 - 1) - Y),
            - 1 * (self.config.C * self.config.H + Y) ** 2 / (2 * self.config.C)
            )

    def grad_g_dual(self, Y):
        return tf.where(
            Y < self.config.C * self.config.H * (self.config.beta - 1),
            (-self.config.beta * self.config.H) * tf.ones_like(Y),
            - Y / self.config.C -  self.config.H
            )

class LQMethod(Method):

    def __init__(self, config, names, graph, monte):
        super().__init__(config, names, graph, monte)
        ones_dd = self.get_ones(self.d, self.d)
        ones_dm = self.get_ones(self.d, self.m)
        ones_md = self.get_ones(self.m, self.d)
        ones_mm = self.get_ones(self.m, self.m)

        ones_d = self.get_ones(self.d)
        ones_m = self.get_ones(self.m)

        self.A = lambda t: ones_dd * self.config.A(t)
        self.B = lambda t: ones_dm * self.config.B(t)
        self.C = lambda t: ones_dd * self.config.C(t)
        self.D = lambda t: ones_dm * self.config.D(t)

        self.Q = lambda t: ones_dd * self.config.Q(t)
        self.R = lambda t: ones_mm * self.config.R(t)
        self.S = lambda t: ones_md * self.config.S(t)

        self.gamma = lambda t: ones_d * self.config.gamma(t)
        self.sigma = lambda t: ones_d * self.config.sigma(t)
        self.p = lambda t: ones_m * self.config.p(t)
        self.q = lambda t: ones_d * self.config.q(t)

        self.G     = ones_dd * self.config.G
        self.g_vec = ones_d  * self.config.g
        
    def fixProcesses(self):

        if 'V1' in self.processes.keys():
            for name in self.processes['V1'].keys():
                for t in range(self.N + 1):
                    v = self.processes['V1'][name][t]
                    self.processes['V1'][name][t] = -1 * v
        if 'Z1' in self.processes.keys():
            for name in self.processes['Z1'].keys():
                for t in range(self.N + 1):
                    v = self.processes['Z1'][name][t]
                    self.processes['Z1'][name][t] = -1 * v
        if 'Gamma1' in self.processes.keys():
            for name in self.processes['Gamma1'].keys():
                for t in range(self.N):
                    v = self.processes['Gamma1'][name][t]
                    self.processes['Gamma1'][name][t] = -1 * v
       

    def restrict(self, a):
        return a


    def drift(self, t, X, a):
        return (
            b_mult(self.A(t), X) 
            + b_mult(self.B(t), a) 
            + self.gamma(t)
            )

    def vol(self, t, X, a):  # d * k
        return (
            self.C(t) @ tf.expand_dims(X, -1) 
            + self.D(t) @ tf.expand_dims(a, -1)
            + tf.expand_dims(self.sigma(t), -1)
            )

    def f(self, t, X, a):
        term1 = b_dot(X, b_mult(self.Q(t), X)) / 2
        term2 = b_dot(a, b_mult(self.S(t), X))
        term3 = b_dot(a, b_mult(self.R(t), a)) / 2
        
        term4 = b_dot(X, self.q(t))
        term5 = b_dot(a, self.p(t))
        
        ret = -1 * (term1 + term2 + term3 + term4 + term5)
        
        if self.config.space == 'whole':
            return ret
        elif self.config.space == 'cone':
            l = tf.where(
                    a > 0,
                    tf.ones_like(a) * 0,
                    a
                    )
            
            
            return ret - 100 *  b_dot(l,l)
      
        else:
            raise NotImplementedError()
              

    def g(self, X):
        return -1 * b_dot(X, b_mult(self.G, X)) / 2 - b_dot(X, self.g_vec)
    
    def grad_g(self,X):
        return -1 * b_mult(self.G, X) - self.g_vec
    
    def grad_H(self, t, X, a, Z, Q):
        return (
            b_mult(tf.linalg.transpose(self.A(t * self.h)), Z) +
            tf.squeeze(tf.linalg.transpose(self.C(t * self.h)) @ Q, -1) -
            b_mult(tf.linalg.transpose(self.S(t * self.h)), a) -
            b_mult(self.Q(t * self.h), X)
            - self.q(t * self.h)
            )
            
    
    
    
    def value_op(self, name, monte):
        return [-1 * super().value_op(name, monte)[0]]



class TemplateMethod(Method):

    def __init__(self, config, names, graph, monte):
        super().__init__(config, names, graph, monte)

    def restrict(self, control):
        return control

    def drift(self, t, X, pi):
        return None

    def vol(self, t, X, pi):  # d *
        return None

    def f(self, t, X, pi):
        return None

    def g(self, X):
        return None

    def dual_restrict(self, control):
        return control

    def drift_dual(self, t, Y, gam):
        return None

    def vol_dual(self, t, Y, gam):  # d *
        return None

    def f_dual(self, t, Y, gam):
        return None

    def g_dual(self, Y):
        return None

def get_method(problem, config, names, graph, monte):
    try:
        return globals()[problem + "Method"](config, names, graph, monte)
    except KeyError:
        raise KeyError("Method for the required problem not found.")

