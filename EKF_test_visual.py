import torch.nn as nn
import torch

from EKF import ExtendedKalmanFilter


def EKFTest(SysModel, test_input, test_target, model_AE_conv, modelKnowledge = 'full', allStates=True):

    N_T = test_target.size()[0]

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')
    
    # MSE [Linear]
    MSE_EKF_linear_arr = torch.empty(N_T)

    EKF = ExtendedKalmanFilter(SysModel, modelKnowledge)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

    KG_array = torch.zeros_like(EKF.KG_array)
    y_test_decoaded = torch.empty([N_T, SysModel.m, SysModel.T_test])
    EKF_out = torch.empty([N_T, SysModel.m, SysModel.T_test])

    for j in range(0, N_T):
        y_mdl_tst = test_input[j, :, :, :]
        for t in range(0, SysModel.T_test):
            AE_input = y_mdl_tst[t, :, :].reshape(1, 1, 24, 24) / 255
            y_test_decoaded[j,:,t] = model_AE_conv(AE_input)

        EKF.GenerateSequence(y_test_decoaded[j, :, :], EKF.T_test)

        if(allStates):
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x, test_target[j, :, :]).item()
        else:
            loc = torch.tensor([True,False,True,False])
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x[loc,:], test_target[j, :, :]).item()
        KG_array = torch.add(EKF.KG_array, KG_array) 
        EKF_out[j,:,:] = EKF.x

    # Average KG_array over Test Examples
    KG_array /= N_T

    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)
    
    print("Extended Kalman Filter - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")

    return [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out]



