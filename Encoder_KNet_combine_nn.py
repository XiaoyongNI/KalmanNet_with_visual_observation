"""# **Class: CNN Encoder + KalmanNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func

from visual_supplementary import decoaded_dimention

# from filing_paths import path_model
# import sys
# sys.path.insert(1, path_model)
# from model import getJacobian

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")

in_mult = 5
out_mult = 40

class Visual_KNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################

    def Build(self, SysModel, encoded_dimention, H_FC = None):

        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n, H_FC)
        self.InitSequence(SysModel.m1x_0)

        # # Number of neurons in the 1st hidden layer
        # H1_KNet = (ssModel.m + ssModel.n) * (10) * 8

        # # Number of neurons in the 2nd hidden layer
        # H2_KNet = (ssModel.m * ssModel.n) * 1 * (4)

        self.InitNet(encoded_dimention, SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitNet(self, encoded_dimention, prior_Q, prior_Sigma, prior_S):
        ######## Init CNN Encoder ########################
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True))

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, encoded_dimention))

        ######## Init KGain Net ########################
        self.seq_len_input = 1
        self.batch_size = 1

        self.prior_Q = prior_Q
        self.prior_Sigma = prior_Sigma
        self.prior_S = prior_S
        


        # GRU to track Q
        self.d_input_Q = self.m * in_mult
        self.d_hidden_Q = self.m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q)
        self.h_Q = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Q).to(dev, non_blocking=True)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * in_mult
        self.d_hidden_Sigma = self.m ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma)
        self.h_Sigma = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).to(dev, non_blocking=True)

        # GRU to track S
        self.d_input_S = self.n ** 2 + 2 * self.n * in_mult
        self.d_hidden_S = self.n ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S)
        self.h_S = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_S).to(dev, non_blocking=True)

        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(
                nn.Linear(self.d_input_FC1, self.d_output_FC1),
                nn.ReLU())

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * out_mult
        self.FC2 = nn.Sequential(
                nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC2, self.d_output_FC2))

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(
                nn.Linear(self.d_input_FC3, self.d_output_FC3),
                nn.ReLU())

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
                nn.Linear(self.d_input_FC4, self.d_output_FC4),
                nn.ReLU())
        
        # Fully connected 5
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * in_mult
        self.FC5 = nn.Sequential(
                nn.Linear(self.d_input_FC5, self.d_output_FC5),
                nn.ReLU())

        # Fully connected 6
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * in_mult
        self.FC6 = nn.Sequential(
                nn.Linear(self.d_input_FC6, self.d_output_FC6),
                nn.ReLU())
        
        # Fully connected 7
        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * in_mult
        self.FC7 = nn.Sequential(
                nn.Linear(self.d_input_FC7, self.d_output_FC7),
                nn.ReLU())

        """
        # Fully connected 8
        self.d_input_FC8 = self.d_hidden_Q
        self.d_output_FC8 = self.d_hidden_Q
        self.d_hidden_FC8 = self.d_hidden_Q * Q_Sigma_mult
        self.FC8 = nn.Sequential(
                nn.Linear(self.d_input_FC8, self.d_hidden_FC8),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC8, self.d_output_FC8))
        """
        #############################################################

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, f, h, m, n, H_FC):

        self.d = decoaded_dimention

        # Set State Evolution Function
        self.f = f
        self.m = m

        # Set Observation Function
        self.H_FC = H_FC
        self.h = h
        self.n = n

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0):
        
        # self.x_out = torch.empty(self.m, T).to(dev, non_blocking=True)

        self.m1x_posterior = M1_0.to(dev, non_blocking=True)
        self.m1x_posterior_previous = self.m1x_posterior.to(dev, non_blocking=True)
        self.m1x_prior_previous = self.m1x_posterior.to(dev, non_blocking=True)
        self.y_previous = self.h(self.m1x_posterior).to(dev, non_blocking=True)

        # KGain saving
        # self.i = 0
        # self.T = T
        # self.KGain_array = self.KG_array = torch.zeros((self.T,self.m,self.n)).to(dev, non_blocking=True)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self, fix_H_flag):
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(self.m1x_posterior)
       
        if fix_H_flag:            
            # Predict the 1-st moment of y
            self.m1y = self.h(self.m1x_prior)
        else:# H is learned FC
            self.H_FC.eval()
            # Predict the 1-st moment of y
            self.m1y = self.H_FC(self.m1x_prior.reshape(1, self.m)).reshape(self.d, 1)

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):

        obs_diff = y - torch.squeeze(self.y_previous)
        obs_innov_diff = y - torch.squeeze(self.m1y)
        fw_evol_diff = torch.squeeze(self.m1x_posterior) - torch.squeeze(self.m1x_posterior_previous)
        fw_update_diff = torch.squeeze(self.m1x_posterior) - torch.squeeze(self.m1x_prior_previous)

        obs_diff = func.normalize(obs_diff, p=2, dim=0, eps=1e-12, out=None)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=0, eps=1e-12, out=None)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=0, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=0, eps=1e-12, out=None)


        # Kalman Gain Network Step
        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)

        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.m, self.n))

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y, fix_H_flag):

        # Compute Priors
        self.step_prior(fix_H_flag)

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Save KGain in array
        # self.KGain_array[self.i] = self.KGain
        # self.i += 1

        # Innovation
        # y_obs = torch.unsqueeze(y, 1)
        dy = y - self.m1y

        # Compute the 1-st posterior moment
        INOV = torch.matmul(self.KGain, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV

        #self.state_process_posterior_0 = self.state_process_prior_0
        self.m1x_prior_previous = self.m1x_prior

        # update y_prev
        self.y_previous = y

        # return
        return torch.squeeze(self.m1x_posterior)

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):

        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1])
            expanded[0, 0, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        ####################
        ### Forward Flow ###
        ####################
        
        # FC 5
        in_FC5 = fw_evol_diff
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU
        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)

        """
        # FC 8
        in_FC8 = out_Q
        out_FC8 = self.FC8(in_FC8)
        """

        # FC 6
        in_FC6 = fw_update_diff
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)


        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)


        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4

        return out_FC2

    ###############
    ### Forward ###
    ###############
    def forward(self, y, fix_H_flag):
        y = y.to(dev, non_blocking=True)
        if len(list(y.size())) >= 2:
            if  len(list(torch.squeeze(y).size()))==0 and self.n==1:
                y = torch.squeeze(y, dim = 1)
            else:
                y = torch.squeeze(y)
        '''
        for t in range(0, self.T):
            self.x_out[:, t] = self.KNet_step(y[:, t])
        '''
        return self.KNet_step(y, fix_H_flag)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(1, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S[0, 0, :] = self.prior_S.flatten()
        hidden = weight.new(1, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma[0, 0, :] = self.prior_Sigma.flatten()
        hidden = weight.new(1, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q[0, 0, :] = self.prior_Q.flatten()



