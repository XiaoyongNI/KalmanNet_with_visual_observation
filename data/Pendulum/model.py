import math
import torch
import numpy as np
from torch import autograd
from PendulumData import Pendulum
from parameters import m, n, H_design, H_mod, transition_noise_q, observation_noise_r

img_size = 24

pend_params = Pendulum.pendulum_default_params()
pend_params[Pendulum.FRICTION_KEY] = 0.1

Pendulum_data = Pendulum(img_size=img_size,
                observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                transition_noise_std = transition_noise_q,
                observation_noise_std = observation_noise_r,
                pendulum_params=pend_params,
                seed=0)

delta_t = Pendulum_data.sim_dt

def f(x):
    c = Pendulum_data.g * Pendulum_data.length * Pendulum_data.mass / Pendulum_data.inertia
    velNew = x[1:2] + delta_t * (c * torch.sin(x[0:1]) - x[1:2] * Pendulum_data.friction)
    x_new = torch.tensor([x[0:1] + delta_t * velNew, velNew])
    return x_new

def h(x):
    return torch.matmul(H_design,x)

def h_add_obs_noise(x):
    if Pendulum_data.observation_noise_std > 0.0:
        observation_noise = torch.from_numpy(Pendulum_data.random.normal(loc=0.0,
                                                scale=Pendulum_data.observation_noise_std,
                                                size=x.shape))
    else:
        observation_noise = torch.zeros(x.shape)
    noisy_state = torch.matmul(H_design,x) + observation_noise

    return noisy_state

def fInacc(x):
    g = 9.81 # Gravitational Acceleration
    L = 1.1 # Radius of pendulum
    result = [x[0]+x[1]*delta_t, x[1]-(g/L * torch.sin(x[0]))*delta_t]
    result = torch.squeeze(torch.tensor(result))
    # print(result.size())
    return result

def hInacc(x):
    return torch.matmul(H_mod,x)
    #return toSpherical(x)

def getJacobian(x, a):
    
    # if(x.size()[1] == 1):
    #     y = torch.reshape((x.T),[x.size()[0]])
    try:
        if(x.size()[1] == 1):
            y = torch.reshape((x.T),[x.size()[0]])
    except:
        y = torch.reshape((x.T),[x.size()[0]])
        
    if(a == 'ObsAcc'):
        g = h
    elif(a == 'ModAcc'):
        g = f
    elif(a == 'ObsInacc'):
        g = hInacc
    elif(a == 'ModInacc'):
        g = fInacc

    Jac = autograd.functional.jacobian(g, y)
    Jac = Jac.view(-1,m)
    return Jac



'''
x = torch.tensor([[1],[1],[1]]).float() 
H = getJacobian(x, 'ObsAcc')
print(H)
print(h(x))

F = getJacobian(x, 'ModAcc')
print(F)
print(f(x))
'''