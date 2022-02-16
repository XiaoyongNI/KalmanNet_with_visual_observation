import torch
torch.set_default_tensor_type(torch.DoubleTensor)
import numpy as np
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from EKF_test import EKFTest
from KalmanNet_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
from Extended_data import N_E, N_CV, N_T
from Pipeline_EKF import Pipeline_EKF
from KalmanNet_nn import KalmanNetNN

from datetime import datetime

from Plot import Plot_extended as Plot

from filing_paths import path_model
import sys
sys.path.insert(1, path_model)
from parameters import T, T_test, m1x_0, m2x_0, m, n, transition_noise_q, observation_noise_r
from model import f, h, h_add_obs_noise

if torch.cuda.is_available():
   cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")


print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

##########################
###  Prepare dataset   ###
##########################
path_results = 'KNet/'
DatafolderName = 'data/Pendulum/' 

# noise parameters
r = torch.tensor([observation_noise_r])
q = torch.tensor([transition_noise_q])

print("1/r2 [dB]: ", 10 * torch.log10(1/r[0]**2))
print("1/q2 [dB]: ", 10 * torch.log10(1/q[0]**2))

# load dataset
dataFile = np.load(DatafolderName+'pdl_data2.npz',encoding = "latin1") 
torchdata = torch.from_numpy(dataFile['Angle']) # convert to torch tensor
torchdata = torch.transpose(torchdata, 1, 2)
print("Data Load")
print("Data size (num_episodes, state dimension, episode_length):",torchdata.size())# dataset in the form of (num_episodes, state dimension, episode_length)

target = torchdata # target is the true state
input = torch.empty_like(target) # input is the noisy observation
for i in range(0,N_T):
   for t in range(0, T_test):
      input[i,:,t] = h_add_obs_noise(target[i,:,t]) # multiply by observation model and add observation noise

# Split into training, Cross-Validation and testing dataset
train_target = target[0:70,:,:]
train_input = input[0:70,:,:]
cv_target = target[70:80,:,:]
cv_input = input[70:80,:,:]
test_target = target[80:100,:,:]
test_input = input[80:100,:,:]
print("trainset size:",train_target.size())
print("cvset size:",cv_target.size())
print("testset size:",test_target.size())
############################
###  KalmanNet and EKF   ###
############################
# Model with full Info
Q_true = (q[0]**2) * torch.eye(m)
R_true = (r[0]**2) * torch.eye(n)
sys_model = SystemModel(f, Q_true, h, R_true, T, T_test)
sys_model.InitSequence(m1x_0, m2x_0)

# Model with partial Info
# Q_mod = (qopt**2) * torch.eye(m)
# R_mod = (r[0]**2) * torch.eye(n)
# sys_model_partialf = SystemModel(fInacc, Q_mod, h, R_mod, T, T_test)
# sys_model_partialf.InitSequence(m1x_0, m2x_0)

## Evaluate EKF true
print("Evaluate EKF true")
[MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model, input, target)
## Evaluate EKF partial (h or r)
# print("Evaluate EKF partial")
# [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(sys_model_partialh, test_input, test_target)
## Evaluate EKF partial optq
# print("Evaluate EKF partial")
# [MSE_EKF_linear_arr_partialoptq, MSE_EKF_linear_avg_partialoptq, MSE_EKF_dB_avg_partialoptq, EKF_KG_array_partialoptq, EKF_out_partialoptq] = EKFTest(sys_model_partialf, test_input, test_target)
# #Evaluate EKF partial optr
# print("Evaluate EKF partial")
# [MSE_EKF_linear_arr_partialoptr, MSE_EKF_linear_avg_partialoptr, MSE_EKF_dB_avg_partialoptr, EKF_KG_array_partialoptr, EKF_out_partialoptr] = EKFTest(sys_model_partialh_optr, test_input, test_target)
## Eval PF partial
# [MSE_PF_linear_arr_partial, MSE_PF_linear_avg_partial, MSE_PF_dB_avg_partial, PF_out_partial, t_PF] = PFTest(sys_model_partialh, test_input, test_target, init_cond=None)
# print(f"MSE PF H NL: {MSE_PF_dB_avg_partial} [dB] (T = {T_test})")


# Save results

# EKFfolderName = 'KNet' + '/'
# torch.save({#'MSE_EKF_linear_arr': MSE_EKF_linear_arr,
# #             'MSE_EKF_dB_avg': MSE_EKF_dB_avg,
#             # 'MSE_EKF_linear_arr_partial': MSE_EKF_linear_arr_partial,
#             # 'MSE_EKF_dB_avg_partial': MSE_EKF_dB_avg_partial,
#             # 'MSE_EKF_linear_arr_partialoptr': MSE_EKF_linear_arr_partialoptr,
#             # 'MSE_EKF_dB_avg_partialoptr': MSE_EKF_dB_avg_partialoptr,
#             }, EKFfolderName+EKFResultName)

# KNet without model mismatch
modelFolder = 'KNet' + '/'
KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
KNet_Pipeline.setssModel(sys_model)
KNet_model = KalmanNetNN()
KNet_model.Build(sys_model)
KNet_Pipeline.setModel(KNet_model)
KNet_Pipeline.setTrainingParams(n_Epochs=100, n_Batch=10, learningRate=1e-4, weightDecay=1e-5)

# KNet_Pipeline.model = torch.load(modelFolder+"model_KNet.pt")

KNet_Pipeline.NNTrain(train_input, train_target, cv_input, cv_target)
[KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(test_input, test_target)
KNet_Pipeline.save()

# KNet with model mismatch
## Build Neural Network
# KNet_model = KalmanNetNN()
# KNet_model.Build(sys_model_partialf)
# # Model = torch.load('KNet/model_KNetNew_DT_procmis_r30q50_T2000.pt',map_location=cuda0)
# ## Train Neural Network
# KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
# KNet_Pipeline.setssModel(sys_model_partialf)
# KNet_Pipeline.setModel(KNet_model)
# KNet_Pipeline.setTrainingParams(n_Epochs=100, n_Batch=10, learningRate=1e-3, weightDecay=1e-6)
# KNet_Pipeline.NNTrain(train_input, train_target, cv_input, cv_target)
# ## Test Neural Network
# [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(test_input, test_target)
# KNet_Pipeline.save()

# # Save trajectories
# # trajfolderName = 'KNet' + '/'
# # DataResultName = traj_resultName[rindex]
# # # EKF_sample = torch.reshape(EKF_out[0,:,:],[1,m,T_test])
# # # EKF_Partial_sample = torch.reshape(EKF_out_partial[0,:,:],[1,m,T_test])
# # # target_sample = torch.reshape(test_target[0,:,:],[1,m,T_test])
# # # input_sample = torch.reshape(test_input[0,:,:],[1,n,T_test])
# # # KNet_sample = torch.reshape(KNet_test[0,:,:],[1,m,T_test])
# # torch.save({
# #             'KNet': KNet_test,
# #             }, trajfolderName+DataResultName)

# ## Save histogram
# EKFfolderName = 'KNet' + '/'
# torch.save({'MSE_EKF_linear_arr': MSE_EKF_linear_arr,
#             'MSE_EKF_dB_avg': MSE_EKF_dB_avg,
#             'MSE_EKF_linear_arr_partial': MSE_EKF_linear_arr_partial,
#             'MSE_EKF_dB_avg_partial': MSE_EKF_dB_avg_partial,
#             # 'MSE_EKF_linear_arr_partialoptr': MSE_EKF_linear_arr_partialoptr,
#             # 'MSE_EKF_dB_avg_partialoptr': MSE_EKF_dB_avg_partialoptr,
#             'KNet_MSE_test_linear_arr': KNet_MSE_test_linear_arr,
#             'KNet_MSE_test_dB_avg': KNet_MSE_test_dB_avg,
#             }, EKFfolderName+EKFResultName)

   





