"""Optimization module"""
import needle as ndl
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            grad = param.grad.detach() + self.weight_decay*param.detach()
            if param in self.u:
                self.u[param] = self.momentum*self.u[param].detach() + (1-self.momentum)*grad.detach()
                param.cached_data = param.cached_data - self.lr*self.u[param].cached_data        
            else:
                self.u[param] = (1-self.momentum)*grad.detach()
                param.cached_data = param.cached_data - self.lr*self.u[param].cached_data
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.u = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            grad = param.grad.detach() + self.weight_decay*param.detach()
            if param in self.u:
                self.u[param] = self.beta1*self.u[param].detach() + (1-self.beta1)*grad.detach()
                self.v[param] = self.beta2*self.v[param].detach() + (1-self.beta2)*(grad**2).detach()
            else:
                self.u[param] = (1-self.beta1)*grad.detach()
                self.v[param] = (1-self.beta2)*(grad**2).detach()
            if (self.t!=0):
                u_correct = self.u[param].detach()/(1-self.beta1**self.t)
                v_correct = self.v[param].detach()/(1-self.beta2**self.t)
                param.cached_data = param.cached_data - (self.lr*u_correct/((v_correct**0.5).detach()+self.eps)).cached_data
            else:
                param.cached_data = param.cached_data - (self.lr*self.u[param]/((self.v[param]**0.5).detach()+self.eps)).cached_data

        ### END YOUR SOLUTION
