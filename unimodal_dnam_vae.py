#!/usr/bin/env python
# coding: utf-8

# In[1]:


# unimodal DNAm variational autoencoder


# In[2]:


import pickle
import pandas as pd


# In[3]:


obj = pd.read_pickle(r'/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Interns_2023/user/TristanD/Data/BLCA_DNAm/TCGA_DNAm_with_survival.pkl')
tcga_dnam = obj.fillna(0)
tcga_dnam = tcga_dnam.drop('Patient ID', axis=1)


# In[4]:


tcga_dnam


# In[5]:


tcga_dnam.fillna(0, inplace=True)


# In[6]:


import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# In[7]:


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


# In[8]:


TCGA_DNAm_Data = TCGA_DNAm_Dataset(tcga_dnam)
print(TCGA_DNAm_Data[0])
print(len(TCGA_DNAm_Data))


# In[9]:


device = torch.device('cuda') if not torch.cuda.is_available() else torch.device('cpu')


# In[10]:


indices = torch.randperm(len(TCGA_DNAm_Data)).tolist()
dataset_train = torch.utils.data.Subset(TCGA_DNAm_Data, indices[:-54])
dataset_val = torch.utils.data.Subset(TCGA_DNAm_Data, indices[-54:-38])
dataset_test = torch.utils.data.Subset(TCGA_DNAm_Data, indices[-38:])


# In[11]:


batch_size = 16


# In[12]:


data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)
data_loader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False, num_workers=4)


# In[13]:


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


# In[14]:


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


# In[15]:


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, input_size):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, input_size)
        self.decoder = Decoder(latent_dims, input_size)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# In[16]:


latent_dims = 256
input_size = 280257 
vae = VariationalAutoencoder(latent_dims, input_size).to(device)

learning_rate = 0.0001
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)


# In[17]:


vae


# In[18]:


checkpoint = torch.load("dnam_vae_50.pth")
vae.load_state_dict(checkpoint["state_dict"])
optimizer.load_state_dict(checkpoint["optimizer"])


# In[19]:


num_epochs = 50


# In[20]:


ftrain_losses = []
fval_losses = []
for epoch in range(num_epochs):
    train_losses = []
    val_losses = []
    vae.train()
    for batch_num, input_data in enumerate(data_loader_train):
        optimizer.zero_grad()
        x, y_time, y_status = input_data
        x = x.to(device).float()
        
        output = vae(x)
        loss = ((x - output)**2).sum() + vae.encoder.kl
        train_losses.append(loss)
        loss.backward()
        optimizer.step()
        if batch_num % 2 == 0:
            print('\tEpoch %d | Batch %d | Loss %6.2f'  % (epoch+51, batch_num, loss))
            
    vae.eval()
    for batch_num, input_data in enumerate(data_loader_val):
        x, y_time, y_status = input_data
        x = x.to(device).float()
        output = vae(x)
        loss = ((x - output)**2).sum() + vae.encoder.kl
        val_losses.append(loss)
    
    ftrain_losses.append(sum(train_losses)/len(train_losses))
    fval_losses.append(sum(val_losses)/len(val_losses))
    
    print('Epoch %d | Loss %6.2f' % (epoch, sum(train_losses)/len(train_losses)))


# In[21]:


import matplotlib.pyplot as plt


# In[22]:


vae.eval()
with torch.no_grad():
    plt.plot(ftrain_losses,label="Training Loss")
    plt.plot(fval_losses,label="Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


# In[23]:


vae.eval()
test_losses = []
for batch_num, input_data in enumerate(data_loader_test):
    x, y_time, y_status = input_data
    x = x.to(device).float()
    output = vae(x)
    loss = ((x - output)**2).sum() + vae.encoder.kl
    test_losses.append(loss)
    
print(sum(test_losses), len(test_losses))


# In[24]:


dnam_vae_v1 = {'model': vae,
          'state_dict': vae.state_dict(),
          'optimizer' : optimizer.state_dict()}

torch.save(dnam_vae_v1, 'dnam_vae_100.pth')


# In[ ]:




