'''
Main file for Encoder CNN + KNet combination training. Dataset is pendulum.
'''

import torch #Machine Learning
from datetime import datetime # getting current time

from EKF_test_visual import EKFTest
from Extended_sysmdl_visual import SystemModel as NL_SystemModel
from KalmanNet_sysmdl import SystemModel as NewArch_SystemModel
from Extended_data_visual import DataGen,DataLoader,DataLoader_GPU, getObs
from Extended_data_visual import N_E, N_CV, N_T,H_fully_connected, H_matrix_for_visual, b_for_visual

from Pipeline_combine import Pipeline_KF

from Encoder_KNet_combine_nn import Visual_KNetNN

from filing_paths import path_model
import sys
sys.path.insert(1, path_model)
from parameters import NL_T, NL_T_test, NL_m1_0, NL_m2_0, NL_m, NL_n
from model import f, h

if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
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
#path_results = 'RTSNet/'


############ Hyper Parameters ##################
learning_rate_list=[1e-5]
weight_decay_list=[1e-5]
fix_H_flag=True
encoded_dimention = 1 # the output dim of encoder
################################################

####################
### Design Model ###
####################
r2 = torch.tensor([1e-4])
# vdB = -20 # ratio v=q2/r2
# v = 10**(vdB/10)
# q2 = torch.mul(v,r2)
q2 = torch.tensor([1e-6]) # can be tuned
print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))
# True model
r = torch.sqrt(r2)
q = torch.sqrt(q2)


sys_model = NL_SystemModel(f, q, h, r, NL_T, NL_T_test, NL_m, NL_n, None)
sys_model.InitSequence(NL_m1_0, NL_m2_0)

Q_mod = q * q * torch.eye(NL_m)
R_mod = r * r * torch.eye(NL_n)
sys_model_KNet = NewArch_SystemModel(f, Q_mod, h, R_mod, NL_T, NL_T_test)
sys_model_KNet.InitSequence(NL_m1_0, NL_m2_0)     

##### Load  H FC Models ##################
h_fully_connected = H_fully_connected(H_matrix_for_visual, b_for_visual)
      
#################################

###################################
### Data Loader (Generate Data) ###
###################################

dataFolderName = 'Simulations/Pendulum' + '/'
dataFileName = 'y24x24_Ttrain30_NE1000_NCV100_NT100_Ttest40_pendulum.pt'
data_name = 'pendulum'

#print("Start Data Gen")
#DataGen(sys_model, dataFolderName + dataFileName, T, T_test,randomInit=False)  # taking time
print("Data Load")
[train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(dataFolderName + dataFileName)

######  Optional: Cut CV set size to speed up training ###########
cv_input = cv_input[0:N_CV,...]
cv_target = cv_target[0:N_CV,...]
##################################################################

# Print out dataset
print("trainset size: x {} y {}".format(train_target.size(),train_input.size()))
print("cvset size: x {} y {}".format(cv_target.size(), cv_input.size()))
print("testset size: x {} y {}".format(test_target.size(), test_input.size()))
# print("trainset dtype: x {} y {}".format(train_target.type(),train_input.type()))
# print("cvset dtype: x {} y {}".format(cv_target.type(), cv_input.type()))
# print("testset dtype: x {} y {}".format(test_target.type(), test_input.type()))


##################
###  KalmanNet ###
##################
print("Start Visual KNet pipeline")
modelFolder = 'KNet' + '/'
KNet_Pipeline = Pipeline_KF(strTime, "KNet", "Visual_KNet", data_name)
KNet_Pipeline.setssModel(sys_model)

KNet_model = Visual_KNetNN()
KNet_model.Build(sys_model_KNet, encoded_dimention)

KNet_Pipeline.setModel(KNet_model)
# check_changs(KNet_Pipeline, model_AE_trained,model_AE_conv_trained, pendulum_data_flag )

for lr in learning_rate_list:
   for wd in weight_decay_list:
      KNet_Pipeline.setTrainingParams(fix_H_flag, n_Epochs=500, n_Batch=10, learningRate=lr, weightDecay=wd)
      #KNet_Pipeline.model = torch.load(modelFolder+"model_KNet.pt")
      title="LR: {} Weight Decay: {} Data {}".format(lr,wd,data_name )
      print(title)
      KNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)
      # check_changs(KNet_Pipeline, model_AE_trained, model_AE_conv_trained, pendulum_data_flag )

#Test
[KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(N_T, test_input, test_target)
KNet_Pipeline.save()




