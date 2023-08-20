#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle


# In[2]:


with open('/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Interns_2023/user/TristanD/Data/BLCA_DNAm/TCGA_DNAm_with_survival.pkl', 'rb') as f:
    tcga_dnam = pickle.load(f)
tcga_dnam = tcga_dnam.fillna(0)
tcga_dnam = tcga_dnam.drop('Patient ID', axis=1)


# In[3]:


tcga_dnam


# In[4]:


tcga_dnam.fillna(0, inplace=True)


# In[5]:


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# In[6]:


class TCGA_DNAm_Dataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        self.dnam_array = dataframe.iloc[:, :-2]
        self.time_array = dataframe['survival_time']
        self.status_array = dataframe['vital_status']
        
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        dnam_array = torch.tensor(self.dnam_array.iloc[idx], dtype=torch.float64)
        time = torch.tensor(self.time_array.iloc[idx], dtype=torch.int32)
        status = torch.tensor(self.status_array.iloc[idx], dtype=torch.int32)
        return (dnam_array, time, status)


# In[7]:


TCGA_DNAm_Data = TCGA_DNAm_Dataset(tcga_dnam)
print(len(TCGA_DNAm_Data))
print(TCGA_DNAm_Data[0])


# In[8]:


device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')


# In[9]:


indices = torch.randperm(len(TCGA_DNAm_Data)).tolist()
dataset_train = torch.utils.data.Subset(TCGA_DNAm_Data, indices[:-54])
dataset_val = torch.utils.data.Subset(TCGA_DNAm_Data, indices[-54:-38])
dataset_test = torch.utils.data.Subset(TCGA_DNAm_Data, indices[-38:])


# In[10]:


batch_size = 16


# In[11]:


data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
data_loader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)


# In[12]:


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, input_size):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_size, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, latent_dims)
        self.linear5 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc
        self.N.scale = self.N.scale
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mu =  self.linear4(x)
        sigma = torch.exp(self.linear5(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z


# In[13]:


class Decoder(nn.Module):
    def __init__(self, latent_dims, input_size):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 2048)
        self.linear4 = nn.Linear(2048, input_size)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        z = torch.sigmoid(self.linear4(z))
        return z


# In[14]:


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, input_size):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, input_size)
        self.decoder = Decoder(latent_dims, input_size)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# In[15]:


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.eval()
    
    return model


# In[16]:


get_ipython().system('nvidia-smi')


# In[17]:


vae = load_checkpoint('dnam_vae_100.pth').to(device)


# In[18]:


vae.state_dict()


# In[19]:


vae.encoder.linear1.weight


# In[20]:


list(vae.parameters())


# In[21]:


class DNAm_MLP(nn.Module):
    def __init__(self, vae, hidden_sizes, output_size):
        super(DNAm_MLP, self).__init__()
        self.encoder = vae.encoder
        self.linear1 = self.encoder.linear1
        self.linear2 = self.encoder.linear2
        self.linear3 = self.encoder.linear3
        self.linear1.weight.requires_grad = False
        self.linear1.bias.requires_grad = False
        self.linear2.weight.requires_grad = False
        self.linear2.bias.requires_grad = False
        self.linear3.weight.requires_grad = False
        self.linear3.bias.requires_grad = False
        self.input_layer = nn.Linear(512, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x


# In[22]:


def cox_loss(survtime, censor, hazard_pred, device):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    #current_batch_len = len(survtime)
    #hazard_pred = hazard_pred.to(device)
    
    current_batch_len = len(censor)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]
    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor.to(device))
    return loss_cox


# In[23]:


hidden_sizes = [256, 128, 64, 32]
output_size = 1
mlp = DNAm_MLP(vae, hidden_sizes, output_size).to(device)

loss_fn = cox_loss

learning_rate = 0.0005
optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)


# In[24]:


mlp.linear1.weight


# In[25]:


num_epochs = 5


# In[26]:


ftrain_losses = []
fval_losses = []
for epoch in range(num_epochs):
    train_losses = []
    val_losses = []
    train_c_indices = []
    val_c_indices = []
    mlp.train()
    for batch_num, input_data in enumerate(data_loader_train):
        optimizer.zero_grad()
        x, y_time, y_status = input_data
        x = x.to(device).float()
        y_time = y_time.to(device).float()
        y_status = y_status.to(device).float()
        
        output = mlp(x)
        loss = loss_fn(y_time, y_status, output, device)
        
#         with torch.no_grad():
#             x, y_time, y_status = input_data
#             x = x.float().to(device)
#             y_time = y_time.float().cpu().numpy()
#             y_status = y_status.float().cpu().numpy()
#             output = mlp(x)
#             c_index = concordance_index(y_time, -output.cpu(), y_status)
#             train_c_indices.append(c_index)
        
        optimizer.zero_grad()
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()
        
        if batch_num % 1 == 0:
            print('\tEpoch %d | Batch %d | Loss %6.2f'  % (epoch, batch_num, loss.item()))
            
    mlp.eval()    
    for batch_num, input_data in enumerate(data_loader_val):
        x, y_time, y_status = input_data
        x = x.to(device).float()
        y_time = y_time.to(device).float()
        y_status = y_status.to(device).float()
        output = mlp(x)
        loss = loss_fn(y_time, y_status, output, device)
        val_losses.append(loss.item())
        
#         with torch.no_grad():
#             x, y_time, y_status = input_data
#             x = x.float().to(device)
#             y_time = y_time.float().cpu().numpy()
#             y_status = y_status.float().cpu().numpy()
#             output = mlp(x)
#             c_index = concordance_index(y_time, -output.cpu(), y_status)
#             val_c_indices.append(c_index)
    
    ftrain_losses.append(sum(train_losses)/len(train_losses))
    fval_losses.append(sum(val_losses)/len(val_losses))
    
    print('Epoch %d | Loss %6.2f' % (epoch, sum(train_losses)/len(train_losses)))


# In[27]:


import matplotlib.pyplot as plt


# In[28]:


mlp.eval()
plt.plot(ftrain_losses,label="training loss")
plt.plot(fval_losses,label="validation loss")
plt.legend()
plt.show()


# In[29]:


from lifelines.utils import concordance_index


# In[30]:


mlp.eval()
c_indices = []
for batch_num, input_data in enumerate(data_loader_test):
    with torch.no_grad():
        x, y_time, y_status = input_data
        x = x.float().to(device)
        y_time = y_time.float()
        y_status = y_status.float()
        output = mlp(x)
        c_index = concordance_index(y_time, -output.cpu(), y_status)
        c_indices.append(c_index)
        print(output, y_time, y_status, c_index)

print()
print(c_indices)
print(sum(c_indices) / len(c_indices))


# In[31]:


dnam_mlp_v2 = {'model': mlp,
          'state_dict': mlp.state_dict(),
          'optimizer' : optimizer.state_dict()}

torch.save(dnam_mlp_v2, 'dnam_mlp_1005.pth')

