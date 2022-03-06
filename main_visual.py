import torch #Machine Learning
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from datetime import datetime # getting current time

from EKF_test_visual import EKFTest
from Linear_sysmdl_visual import SystemModel
from Extended_sysmdl_visual import SystemModel as NL_SystemModel
from KalmanNet_sysmdl import SystemModel as NewArch_SystemModel
from Extended_data_visual import DataGen,DataLoader,DataLoader_GPU, getObs
from Extended_data_visual import N_E, N_CV, N_T, F, F_rotated, T, T_test, m1_0, m2_0, m, n,H_fully_connected, H_matrix_for_visual, b_for_visual
from visual_supplementary import y_size, check_changs
from Pipeline_KF_visual import Pipeline_KF

from KalmanNet_nn_LinearCase_OldArch_visual import KalmanNetNN
from KalmanNet_nn_OldArch_visual import KalmanNetNN as Extended_KalmanNetNN
from KalmanNet_nn_NewArch_visual import KalmanNetNN as KalmanNetNN_NewArch

from main_AE import Autoencoder, Encoder

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
learning_rate_list=[1e-4]
weight_decay_list=[1e-5]
fix_H_flag=True
pendulum_data_flag=True # true for pendulum data, false for linear synthetic data
encoded_dimention = 1 # the output dim of encoder
matrix_data_flag = True # true for data in matrix form, false for data in image form
old_arch_flag = False # true for old architecture of KNet, false for new. (architecture ref: KNet_TSP)
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

if pendulum_data_flag:
   sys_model = NL_SystemModel(f, q, h, r, NL_T, NL_T_test, NL_m, NL_n, None)
   sys_model.InitSequence(NL_m1_0, NL_m2_0)
   if not old_arch_flag:
      Q_mod = q * q * torch.eye(NL_m)
      R_mod = r * r * torch.eye(NL_n)
      sys_model_KNet = NewArch_SystemModel(f, Q_mod, h, R_mod, T, T_test)
      sys_model_KNet.InitSequence(NL_m1_0, NL_m2_0)     
else:
   sys_model = SystemModel(F, q, H_matrix_for_visual, r, T, T_test)
   sys_model.InitSequence(m1_0, m2_0)

##### Load  Encoder Models ##################
h_fully_connected = H_fully_connected(H_matrix_for_visual, b_for_visual)

if matrix_data_flag:
   model_AE_trained = None      
   model_AE_conv_trained = None
else:
   model_AE_trained = Autoencoder()
   model_AE_trained.load_state_dict(torch.load('saved_models/AE_model_syntetic.pt'))
   model_AE_conv_trained = Encoder(encoded_dimention)
   model_AE_conv_trained.load_state_dict(torch.load('saved_models/Only_conv_encoder.pt'))
      
#################################

# Mismatched model
#sys_model_partialh = SystemModel(F, q, H_wrong_visual_function, r, T, T_test)
#sys_model_partialh.InitSequence(m1_0, m2_0)

###################################
### Data Loader (Generate Data) ###
###################################
if pendulum_data_flag:
   dataFolderName = 'Simulations/Pendulum' + '/'
   dataFileName = 'y24x24_Ttrain30_NE1000_NCV100_NT100_Ttest40_pendulum.pt'
   data_name = 'pendulum'
else:
   dataFolderName = 'Simulations/Synthetic_visual' + '/'
   dataFileName = 'y{}x{}_Ttrain{}_NE{}_NCV{}_NT{}_Ttest{}_Sigmoid.pt'.format(y_size,y_size, T,N_E,N_CV,N_T, T_test)
   data_name = 'syntetic'
#print("Start Data Gen")
#DataGen(sys_model, dataFolderName + dataFileName, T, T_test,randomInit=False)  # taking time
print("Data Load")
[train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(dataFolderName + dataFileName)

######  Optional: Cut CV set size to enhance training speed ######
cv_input = cv_input[0:N_CV,...]
cv_target = cv_target[0:N_CV,...]
##################################################################
if matrix_data_flag:
   # Observations: y = h(x) + R
   train_input = getObs(train_target,h,N_E,NL_n,NL_T)  
   train_input = train_input + torch.randn_like(train_input) * r

   cv_input = getObs(cv_target,h,N_CV,NL_n,NL_T)  
   cv_input = cv_input + torch.randn_like(cv_input) * r

   test_input = getObs(test_target,h,N_T,NL_n,NL_T_test)  
   test_input = test_input + torch.randn_like(test_input) * r

# Print out dataset
print("trainset size: x {} y {}".format(train_target.size(),train_input.size()))
print("cvset size: x {} y {}".format(cv_target.size(), cv_input.size()))
print("testset size: x {} y {}".format(test_target.size(), test_input.size()))
# print("trainset dtype: x {} y {}".format(train_target.type(),train_input.type()))
# print("cvset dtype: x {} y {}".format(cv_target.type(), cv_input.type()))
# print("testset dtype: x {} y {}".format(test_target.type(), test_input.type()))

##############################
### Evaluate Kalman Filter ###
##############################
if pendulum_data_flag:
   print("Evaluate Kalman Filter True")
   [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out] = EKFTest(sys_model, test_input, test_target, model_AE_conv_trained, matrix_data_flag)
   #print("Evaluate Kalman Filter Partial")
   #[MSE_KF_linear_arr_partialh, MSE_KF_linear_avg_partialh, MSE_KF_dB_avg_partialh] = KFTest(sys_model_partialh, test_input, test_target)

# DatafolderName = 'Data' + '/'
# DataResultName = '10x10_Ttest1000' 
# torch.save({
#             'MSE_KF_linear_arr': MSE_KF_linear_arr,
#             'MSE_KF_dB_avg': MSE_KF_dB_avg,
#             'MSE_RTS_linear_arr': MSE_RTS_linear_arr,
#             'MSE_RTS_dB_avg': MSE_RTS_dB_avg,
#             }, DatafolderName+DataResultName)

##################
###  KalmanNet ###
##################
print("Start KNet pipeline")
modelFolder = 'KNet' + '/'
KNet_Pipeline = Pipeline_KF(strTime, "KNet", "KalmanNet", data_name)
KNet_Pipeline.setssModel(sys_model)

if pendulum_data_flag:
   if old_arch_flag:
      KNet_model = Extended_KalmanNetNN()
      KNet_model.Build(sys_model)
   else:
      KNet_model = KalmanNetNN()
      KNet_model = KalmanNetNN_NewArch()
      KNet_model.Build(sys_model_KNet)
else:
   KNet_model = KalmanNetNN()
   KNet_model.Build(sys_model,h_fully_connected)
KNet_Pipeline.setModel(KNet_model)
# check_changs(KNet_Pipeline, model_AE_trained,model_AE_conv_trained, pendulum_data_flag )

for lr in learning_rate_list:
   for wd in weight_decay_list:
      KNet_Pipeline.setTrainingParams(fix_H_flag, pendulum_data_flag, n_Epochs=500, n_Batch=10, learningRate=lr, weightDecay=wd)
      #KNet_Pipeline.model = torch.load(modelFolder+"model_KNet.pt")
      title="LR: {} Weight Decay: {} Data {}".format(lr,wd,data_name )
      print(title)
      KNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target, title, model_AE_trained, model_AE_conv_trained, matrix_data_flag)
      # check_changs(KNet_Pipeline, model_AE_trained, model_AE_conv_trained, pendulum_data_flag )

#Test
[KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(N_T, test_input, test_target, model_AE_trained, model_AE_conv_trained, matrix_data_flag)
KNet_Pipeline.save()




