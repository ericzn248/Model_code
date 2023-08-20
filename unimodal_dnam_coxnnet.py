#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


# In[2]:


obj = pd.read_pickle('/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Interns_2023/user/TristanD/Data/BLCA_DNAm/TCGA_DNAm_with_survival.pkl')
tcga_dnam = obj.fillna(0)
tcga_dnam = tcga_dnam.drop('Patient ID', axis=1)
# tcga_dnam['vital_status'].replace(0, True, inplace = True)
# tcga_dnam['vital_status'].replace(1, False, inplace = True)
tcga_dnam


# In[3]:


# class TCGA_DNAm_Dataset(Dataset):
#     def __init__(self, dataframe):
#         self.df = dataframe
    
#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, idx):
#         dnam_array = torch.tensor(self.df.iloc[idx], dtype=torch.float64)
#         return dnam_array


# In[4]:


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
        status = torch.tensor(self.status_array.iloc[idx], dtype=bool)
        return (dnam_array, time, status)


# In[5]:


TCGA_DNAm_Data = TCGA_DNAm_Dataset(tcga_dnam)


# In[6]:


TCGA_DNAm_Data[0]


# In[24]:


device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
device


# In[25]:


indices = torch.randperm(len(TCGA_DNAm_Data)).tolist()
dataset_train = torch.utils.data.Subset(TCGA_DNAm_Data, indices[:-100])
dataset_val = torch.utils.data.Subset(TCGA_DNAm_Data, indices[-100:-50])
dataset_test = torch.utils.data.Subset(TCGA_DNAm_Data, indices[-50:])


# In[26]:


batch_size = 25


# In[27]:


data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
data_loader_val = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
data_loader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)


# In[28]:


class DNAm_MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DNAm_MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)
        return x


# In[29]:


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


# In[30]:


input_size = 280257
hidden_sizes = [512, 256, 128, 64, 32]
output_size = 1
model = DNAm_MLP(input_size, hidden_sizes, output_size).to(device)

loss_fn = cox_loss

learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# In[31]:


model


# In[32]:


num_epochs = 64


# In[33]:


# Set the start method for multiprocessing to 'spawn'
# mp.set_start_method('spawn')
# print(mp.get_start_method(allow_none =True))

# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(data_loader_train), epochs=num_epochs)

model.train()
ftrain_losses = []
ftest_losses = []
ftrain_c_indices = []
ftest_c_indices = []
for epoch in range(num_epochs):
    train_losses = []
    test_losses = []
    train_c_indices = []
    test_c_indices = []
    for batch_num, input_data in enumerate(data_loader_train):
        optimizer.zero_grad()
        
        x, y_time, y_status = input_data
        x = x.to(device).float()
        y_time = y_time.to(device).float()
        y_status = y_status.to(device).float()
        output = model(x)
        loss = loss_fn(y_time, y_status, output, device)
        train_losses.append(loss.item())
        
        with torch.no_grad():
            x, y_time, y_status = input_data
            x = x.float().to(device)
            y_time = y_time.float().cpu().numpy()
            y_status = y_status.float().cpu().numpy()
            output = model(x)
            c_index = concordance_index(y_time, -output.cpu(), y_status)
            train_c_indices.append(c_index)

        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        if batch_num % 1 == 0: print('\tEpoch %d | Batch %d | Loss %6.2f'  % (epoch, batch_num, loss.item()))
        
    for batch_num, input_data in enumerate(data_loader_val):
        x, y_time, y_status = input_data
        x = x.to(device).float()
        y_time = y_time.to(device).float()
        y_status = y_status.to(device).float()
        output = model(x)
        loss = loss_fn(y_time, y_status, output, device)
        test_losses.append(loss.item())
        
        with torch.no_grad():
            x, y_time, y_status = input_data
            x = x.float().to(device)
            y_time = y_time.float().cpu().numpy()
            y_status = y_status.float().cpu().numpy()
            output = model(x)
            c_index = concordance_index(y_time, -output.cpu(), y_status)
            test_c_indices.append(c_index)
            
    ftrain_losses.append(sum(train_losses)/len(train_losses))
    ftest_losses.append(sum(test_losses)/len(test_losses))
    
    ftrain_c_indices.append(sum(train_c_indices)/len(train_c_indices))
    ftest_c_indices.append(sum(test_c_indices)/len(test_c_indices))

    
    print('Epoch %d | Loss %6.2f' % (epoch, sum(train_losses)/len(train_losses)))


# In[34]:


# from lifelines.utils import concordance_index

# model.train()
# ftrain_losses = []
# ftest_losses = []
# all_c_index_vals = []
# for epoch in range(num_epochs):
#     train_losses = []
#     test_losses = []
#     for batch_num, input_data in enumerate(data_loader_train):
#         optimizer.zero_grad()
#         x, y_time, y_status = input_data
#         output = model(x)
#         print(y_time, type(y_time))
#         y_timea = y_time.detach().numpy()
#         print(y_timea, type(y_timea))
#         outputa = output.detach.numpy()
#         y_statusa = y_status.detach().numpy()
#         c_index = concordance_index(y_timea, outputa, y_statusa)
#         all_c_index_vals.append(c_index)
#         print(c_index)
        
#         x = x.to(device).float()
#         y_time = y_time.to(device).float()
#         y_status = y_status.to(device).float()
#         print(y_time, type(y_time))
        
#         output = model(x)
#         loss = loss_fn(y_time, y_status, output, device)
#         optimizer.zero_grad()
#         loss.backward()
#         train_losses.append(loss.item())
#         optimizer.step()
        
        
        
#         if batch_num % 1 == 0:
#             print('\tEpoch %d | Batch %d | Loss %6.2f'  % (epoch, batch_num, loss.item()))
        
#     for batch_num, input_data in enumerate(data_loader_test): #changed from train: may be a source of error
#         #print(len(input_data))
#         x, y_time, y_status = input_data
#         x = x.to(device).float()
#         y_time = y_time.to(device).float()
#         y_status = y_status.to(device).float()
#         output = model(x)
               
#         loss = loss_fn(y_time, y_status, output, device)
#         test_losses.append(loss.item())
        
    
    
#     ftrain_losses.append(sum(train_losses)/len(train_losses))
#     ftest_losses.append(sum(test_losses)/len(test_losses))
    
#     print('Epoch %d | Loss %6.2f' % (epoch, sum(train_losses)/len(train_losses)))


# In[35]:


import matplotlib.pyplot as plt
plt.plot(ftrain_losses, label='Training Loss')
plt.plot(ftest_losses, label='Testing Loss')
# Add labels, title, and legend
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.show()


# In[36]:


plt.plot(ftrain_c_indices, label='Train C Indices')
plt.plot(ftest_c_indices, label='Test C Indices')
plt.xlabel('Epochs')
plt.ylabel('C Index')
plt.title('Training and Testing C Index')
plt.legend()
plt.show()


# ![image-3.png](attachment:image-3.png)
# ![image-4.png](attachment:image-4.png)
# ![image-5.png](attachment:image-5.png)
# 
# ![image-6.png](attachment:image-6.png)
# ![image-7.png](attachment:image-7.png)
# 
# C_index = 0.6815561959654178
# 
# 
# Max C_index = 0.7305475504322767

# In[37]:


print(max(ftest_c_indices))


# In[39]:


from lifelines.utils import concordance_index

model.eval()
c_indices = []
for batch_num, input_data in enumerate(data_loader_test):
    print(batch_num, input_data)
    with torch.no_grad():
        x, y_time, y_status = input_data
#         print(type(x), type(y_time), type(y_status))
        x = x.float().to(device)
        y_time = y_time.float().cpu().numpy()
        y_status = y_status.float().cpu().numpy()
        output = model(x).cpu()
        print(output, y_time, y_status)
        output = output[:, ~torch.isnan(output).any(dim=0)]
        print(output)
        c_index = concordance_index(y_time, -output, y_status)
        c_indices.append(c_index)
        print(output, y_time, y_status, c_index)

print()
# print(c_indices)
print(sum(c_indices) / len(c_indices))


# In[21]:


torch.save(model.state_dict(), "model(.4).pth")

# How to load a model

import torch

#Create a model architecture
model = YourModelClass()  # Define your model architecture

#Load saved parameters
model.load_state_dict(torch.load("model.pth"))
model.eval()  # Put the model in evaluation mode
# In[ ]:




