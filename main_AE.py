'''
The file defines the FC version of Auto-Encoder and the CNN version of Auto-Encoder,
their training process,
and testing
'''

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Extended_data import N_E, N_CV, N_T, T, T_test
from Extended_data import DataLoader_GPU
import numpy as np

import sys
sys.path.insert(1, 'data/Synthetic_visual/')
from visual_supplementary import create_dataset, visualize_similarity, y_size

def train_conv(encoder, decoder, train_loader, val_loader, loss_fn, optimizer, num_epochs,flag_only_encoder):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss_epoch=[]
    val_loss_epoch = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for epoch in range(num_epochs):
        train_loss = []
        for k,batch in enumerate(train_loader):  # with "_" we just ignore the labels (the second element of the dataloader tuple)
            if flag_only_encoder:
                image_batch=batch[0]
                targets_batch=batch[1]
            else:
                image_batch=batch[0]
                targets_batch=batch[0]
            if(image_batch.shape[0]==256):
                encoded_data = encoder(image_batch.reshape(256,1,24,24)/255)
                if flag_only_encoder:
                    decoded_data = encoded_data
                    loss = loss_fn(decoded_data.float(), targets_batch.float())
                else:
                    decoded_data = decoder(encoded_data)
                    if(k==10):
                        check_learning_process(targets_batch.reshape(256,1,24,24)/255, decoded_data, epoch, 'train')
                    loss = loss_fn(decoded_data, targets_batch.reshape(256, 1, 24, 24) / 255)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Print batch loss
                train_loss.append(loss.detach().cpu().numpy())
        train_loss_epoch.append(np.mean(train_loss))
        epoch_val_loss=test_epoch(encoder, decoder, val_loader, loss_fn, epoch, flag_only_encoder)
        val_loss_epoch.append(epoch_val_loss)
        print('epoch: {} train loss: {} val loss: {}'.format(epoch,np.mean(train_loss),epoch_val_loss))
    return encoder, decoder, val_loss_epoch, train_loss_epoch

def test_epoch(encoder, decoder, val_loader, loss_fn, epoch,  flag_only_encoder):
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        val_loss = []
        for k,batch in enumerate(val_loader):
            image_batch=batch[0]
            targets_batch=batch[1]
            if (image_batch.shape[0] == 256): #taking only full batch
                encoded_data = encoder(image_batch.reshape(256, 1, 24, 24) / 255)
                if flag_only_encoder:
                    decoded_data = encoded_data
                    loss = loss_fn(decoded_data.float(), targets_batch.float())
                else:
                    decoded_data = decoder(encoded_data)
                    if (k == 10):
                        check_learning_process(image_batch.reshape(256, 1, 24, 24) / 255, decoded_data, epoch, 'val')
                    loss = loss_fn(decoded_data, targets_batch.reshape(256, 1, 24, 24) / 255)
                val_loss.append(loss.detach().cpu().numpy())
    return np.mean(val_loss)

def train(model, H_data_train,H_data_valid, num_epochs, batch_size, learning_rate):
    torch.manual_seed(42)
    #criterion_x = nn.MSELoss()
    criterion_y = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_loader = torch.utils.data.DataLoader(H_data_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(H_data_valid, batch_size=len(H_data_valid), shuffle=True)
    outputs = []
    epoch_list = []
    Loss_train_list = []
    Loss_valid_list = []
    for epoch in range(num_epochs):
        for data in train_loader:
            state_batch, img_batch = data
            x_bottom_batch,recon_batch = model(img_batch)
            #check_learning_process(img_batch, recon_batch, epoch, 'Train')
            loss_y = criterion_y(recon_batch, img_batch)
            #loss_x = criterion_x(x_bottom_batch, state_batch)
            #Total_loss = loss_y + loss_x
            loss_y.backward()
            optimizer.step()
            optimizer.zero_grad()

            #validation
            for data in val_loader:
                state_batch, img_batch = data
                x_bottom_batch, recon_batch = model(img_batch)
                #check_learning_process(img_batch, recon_batch, epoch, 'Val')
                loss_y_valid = criterion_y(recon_batch, img_batch)

        epoch_list.append(epoch)
        Loss_valid_list.append(float(loss_y_valid))
        Loss_train_list.append(float(loss_y))
        print('Epoch:{}, Train Loss:{:.7f}, Valid Loss:{:.7f}'.format(epoch+1, float(loss_y), float(loss_y_valid)))
    outputs.append((state_batch, x_bottom_batch, img_batch, recon_batch))
    return outputs, epoch_list, Loss_train_list, Loss_valid_list, model

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            torch.nn.Linear(y_size * y_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 5)
        )
        self.decoder = nn.Sequential(
            torch.nn.Linear(5, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, y_size * y_size),
            torch.nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x_bottom = x
        y_rec = self.decoder(x_bottom)
        return x_bottom,y_rec

class Encoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
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
            nn.Linear(32, encoded_space_dim))

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, 128),
            nn.ReLU(True))
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=(32, 2, 2))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=0, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(8, 1, 3, stride=2,padding=1, output_padding=1))

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

def check_learning_process(img_batch,recon_batch,epoch, name):
    y_nump = img_batch[32].reshape(24,24).detach().numpy().squeeze()
    y_recon_nump = recon_batch[32].reshape(24,24).detach().numpy().squeeze()
    #y_nump = img_batch[10][5]
    #y_recon_nump = recon_batch[10][5]
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 2, 1)
    plt.imshow(y_nump, cmap='gray')
    plt.axis('off')
    plt.title("origin")

    fig.add_subplot(1, 2, 2)
    plt.imshow(y_recon_nump, cmap='gray')
    plt.axis('off')
    plt.title("reconstruct")
    fig.savefig('AE_Process/{} Process at epoch {}.PNG'.format(name, epoch))

def train_AE_FC():
    dataFolderName = 'Simulations/Synthetic_visual' + '/'
    dataFileName = 'y{}x{}_Ttrain{}_NE{}_NCV{}_NT{}_Ttest{}_Sigmoid.pt'.format(y_size,y_size, T,N_E,N_CV,N_T,T_test)
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(dataFolderName + dataFileName)
    H_data_train = create_dataset(train_input, train_target, N_E)
    H_data_valid = create_dataset(cv_input, cv_target, N_CV)
    model = Autoencoder()
    max_epochs = 20
    BATCH_ZISE = 128
    LR = 1e-3
    outputs, epoch_list, Loss_train_list, Loss_valid_list, AE_model = train(model, H_data_train,H_data_valid, num_epochs=max_epochs, batch_size=BATCH_ZISE, learning_rate=LR)
    return AE_model

    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 1, 1)
    plt.plot(epoch_list, Loss_train_list, label='Train')
    plt.plot(epoch_list, Loss_valid_list, label='Val')
    plt.title("AutoEncoader Loss")
    fig.savefig('AE_Process/Loss')

    visualize_similarity(outputs[0][2][22].squeeze().detach().numpy().reshape((y_size,y_size)), outputs[0][3][22].squeeze().detach().numpy().reshape((y_size,y_size)))

def print_process(val_loss_list, train_loss_list , flag_only_encoder):
    if flag_only_encoder:
        title='only encoder'
    else:
        title='Auto Encoder'
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 1, 1)
    plt.plot(train_loss_list, label='train')
    plt.plot(val_loss_list, label='val')
    plt.title("Loss of Auto Encoder")
    plt.legend()
    plt.savefig(r"./AE_Process/Learning_process {} ".format(title))

def load_pendulum_npz(path):
    file = np.load(path)
    file_np = file.f.images
    return file_np

def training_AE_conv(flag_only_encoder):
    imgs_np = load_pendulum_npz(r'./data/Pendulum/imgs.npz')
    # noisy_imgs_np = load_pendulum_npz(r'./data/Pendulum/noisy_imgs.npz')
    states_np = load_pendulum_npz(r'./data/Pendulum/states.npz')

    #check_learning_process(imgs_np, noisy_imgs_np, 1, 1)

    train_list=[]
    for k in range(imgs_np.shape[0]):
        sample=imgs_np[k]
        for t in range(sample.shape[0]):
            img=sample[t]
            train_list.append(img)

    if flag_only_encoder:
        chosen_targets_np=states_np
    else:
        chosen_targets_np = imgs_np

    targets_list=[]
    for k in range(chosen_targets_np.shape[0]):# number of sequences
        sample=chosen_targets_np[k]
        for t in range(sample.shape[0]): # sequence length
            target=sample[t,0:1] #learn only the angle, since angular velocity is hard to learn from single image
            targets_list.append(target)

    lr = 0.0002
    torch.manual_seed(0)
    d = 1 #learn only the angle
    num_epochs=30
    batch_size=256
    
    # Split into train, cross-validation and test dataset
    data_train = train_list[22500:]
    targets_train = targets_list[22500:]
    dataset_train=[]
    data_val_test = train_list[:22500]
    targets_val_test = targets_list[:22500]

    data_val = data_val_test[:10000]
    targets_val = targets_val_test[:10000]
    dataset_val = []
    data_test = data_val_test[10000:]
    targets_test = targets_val_test[10000:]
    dataset_test = []

    for k in range(len(data_train)):
        dataset_train.append((data_train[k],targets_train[k]))
    for k in range(len(data_val)):
        dataset_val.append((data_val[k],targets_val[k]))
    for k in range(len(data_test)):
        dataset_test.append((data_test[k], targets_test[k]))

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)

    loss_fn = torch.nn.MSELoss()
    encoder = Encoder(encoded_space_dim=d, fc2_input_dim=128)
    decoder = Decoder(encoded_space_dim=d, fc2_input_dim=128)
    params_to_optimize = [{'params': encoder.parameters()},{'params': decoder.parameters()}]
    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)
    encoder, decoder, val_loss_epoch, train_loss_epoch = train_conv(encoder, decoder, train_loader,valid_loader, loss_fn, optim, num_epochs,flag_only_encoder)
    print_process(val_loss_epoch, train_loss_epoch, flag_only_encoder)
    return encoder, decoder, test_loader

if __name__ == '__main__':
    #train_AE_FC()
    flag_only_encoder=True
    encoder, decoder, test_loader = training_AE_conv(flag_only_encoder)
    if flag_only_encoder:
        torch.save(encoder, r"./saved_models/Only_conv_encoder.pt")
    else:
        torch.save(encoder, r"./saved_models/AutoEncoder_conv_encoder.pt")
        torch.save(decoder, r"./saved_models/AutoEncoder_conv_decoder.pt")

    encoder_loaded = torch.load(r"./saved_models/AutoEncoder_conv_encoder.pt")
    decoder_loaded = torch.load(r"./saved_models/AutoEncoder_conv_decoder.pt")
    test_loss = test_epoch(encoder_loaded, decoder_loaded, test_loader, torch.nn.MSELoss(), 1, flag_only_encoder)
    print("Test loss is: {}".format(test_loss))






