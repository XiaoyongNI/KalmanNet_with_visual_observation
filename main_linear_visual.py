import torch #Machine Learning
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from datetime import datetime # getting current time

from Linear_sysmdl_visual import SystemModel
from Extended_data_visual import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
from Extended_data_visual import N_E, N_CV, N_T, F, F_rotated, T, T_test, m1_0, m2_0, m, n,H_fully_connected, H_matrix_for_visual, b_for_visual
from visual_supplementary import y_size, check_changs
from Pipeline_KF_visual import Pipeline_KF
from KalmanNet_nn_visual import KalmanNetNN
from main_AE import Autoencoder, Encoder

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

####################
### Design Model ###
####################
r2 = torch.tensor([1.])
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)
print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))
# True model
r = torch.sqrt(r2)
q = torch.sqrt(q2)
sys_model = SystemModel(F, q, H_matrix_for_visual, r, T, T_test)
sys_model.InitSequence(m1_0, m2_0)

############ Hyper Parameters ##################
learning_rate_list=[2e-5]
weight_decay_list=[1e-4]
fix_H_flag=True
pendulum_data_flag=True
encoded_dimention = 1
################################################

##### Load  Encoder Models ##################
h_fully_connected = H_fully_connected(H_matrix_for_visual, b_for_visual)
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
print("trainset size: x {} y {}".format(train_target.size(),train_input.size()))
print("cvset size: x {} y {}".format(cv_target.size(), cv_input.size()))
print("testset size: x {} y {}".format(test_target.size(), test_input.size()))

##############################
### Evaluate Kalman Filter ###
##############################
#print("Evaluate Kalman Filter True")
#[MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(sys_model, test_input, test_target)
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

KNet_model = KalmanNetNN()
KNet_model.Build(sys_model,h_fully_connected)
KNet_Pipeline.setModel(KNet_model)
check_changs(KNet_Pipeline, model_AE_trained,model_AE_conv_trained, pendulum_data_flag )

for lr in learning_rate_list:
   for wd in weight_decay_list:
      KNet_Pipeline.setTrainingParams(fix_H_flag, pendulum_data_flag, n_Epochs=300, n_Batch=64, learningRate=lr, weightDecay=wd)
      #KNet_Pipeline.model = torch.load(modelFolder+"model_KNet.pt")
      title="LR: {} Weight Decay: {} Data {}".format(lr,wd,data_name )
      print(title)
      KNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target, title, model_AE_trained, model_AE_conv_trained)
      check_changs(KNet_Pipeline, model_AE_trained, model_AE_conv_trained, pendulum_data_flag )

#Test
[KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(N_T, test_input, test_target, model_AE_trained, model_AE_conv_trained)
KNet_Pipeline.save()




