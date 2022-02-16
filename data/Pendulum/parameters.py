import math
import numpy as np
import torch
torch.set_default_tensor_type(torch.DoubleTensor)

#########################
### Design Parameters ###
#########################
m = 2 # dimension of state
n = 2 # dimension of observation

# initial angle of pendulum is a uniform distribution 
m1x_0 = torch.tensor([0.5*(1.5*np.pi+0.5*np.pi),0]) # initial mean angle and initial velocity
variance = 1/12 * (1.5*np.pi - 0.5*np.pi)**2 # variance of uniform distribution
m2x_0 = torch.tensor([[variance,0],[0,0]])

##########################################
### Generative Parameters For Pendulum ###
##########################################
# Length of Time Series Sequence
T = 75
T_test = 75

H_design = torch.eye(m)

# Noise Parameters
transition_noise_q = 0.1
observation_noise_r = 1e-2

#################################################
#### Rotated Transition or Observation model ####
#################################################

## Rotated Observation H
alpha_degree = 1
rotate_alpha = torch.tensor([alpha_degree/180*math.pi])
cos_alpha = torch.cos(rotate_alpha)
sin_alpha = torch.sin(rotate_alpha)
rotate_matrix = torch.tensor([[cos_alpha, -sin_alpha],
                          [sin_alpha, cos_alpha]])
# print(rotate_matrix)
# F_rotated = torch.mm(F,rotate_matrix) #inaccurate process model
H_mod = torch.matmul(H_design,rotate_matrix) #inaccurate observation model
