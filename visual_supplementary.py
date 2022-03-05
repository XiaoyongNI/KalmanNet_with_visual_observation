import numpy as np
import math
import torch
from Extended_data_visual import T, m, n
import matplotlib.pyplot as plt

## constants
x_size = 2
y_size = 28
decoaded_dimention = 5

def H_visual_function_for_creating_data(x_sample):
    y_sample = torch.zeros(y_size, y_size)
    x_1 = x_sample[0]
    x_2 = x_sample[1]
    for i in range(y_size):
        for j in range(y_size):
            y_sample[i, j] = math.sin(x_1 * math.sin(i / (math.pow(2, (j / 2))))) + math.cos(x_2 * math.cos(i / (math.pow(2, (j / 2)))))
    return torch.sigmoid(y_sample)

def H_wrong_visual_function(x_sample):
    return torch.zeros((y_size, y_size))

def create_dataset(train_input, train_target, size):
    dataset=[]
    for n in range (size):
        for t in range(T):
            x_sample = train_target[n,:,t].reshape((m,1))
            y_sample = train_input[n,:,t].reshape((1,y_size*y_size))
            #y_nump = y_sample.reshape((1,y_size,y_size)).cpu().detach().squeeze().numpy()
            #plt.imshow(y_nump)
            dataset.append((x_sample,y_sample))
    return dataset

def visualize_similarity(y, y_rec):
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 3, 1)
    plt.imshow(y)
    plt.axis('off')
    plt.title("origin")

    fig.add_subplot(1, 3, 2)
    plt.imshow(y_rec)
    plt.axis('off')
    plt.title("reconstruct")

    fig.add_subplot(1, 3, 3)
    plt.imshow(y-y_rec)
    plt.axis('off')
    plt.title("difference")
    fig.savefig('AE Process/Difference')

    #loss = torch.nn.MSELoss()
    #print("MSE between images: {}".format(loss(y,y_rec)))
    Itay=28


def check_changs(KNet_Pipeline):
    print("AE weights after training :")
    print(KNet_Pipeline.model.model_AE.state_dict())
    print("H weights after training :")
    print(KNet_Pipeline.model.H_FC.state_dict())
    print("Kgain weights after training :")
    print(KNet_Pipeline.model.state_dict())



