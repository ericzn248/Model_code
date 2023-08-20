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

from lifelines.utils import concordance_index


# In[2]:


with open('/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Interns_2023/user/TristanD/Data/BLCA_HiTIMED/BLCA_Deconv6_with_survival.pkl', 'rb') as handle:
    tcga_hitimed = pickle.load(handle) # loading the final pickle file
tcga_hitimed = tcga_hitimed[tcga_hitimed['vital_status'].notna()] # removes rows where the vital_status column has NaN values
tcga_hitimed = tcga_hitimed.drop(columns=['submitter_id'])

# tcga_hitimed['vital_status'] = tcga_hitimed['vital_status'].apply(lambda x: 1 if x == 0 else 0)

tcga_hitimed


# In[3]:


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
device


# In[4]:


class TCGA_HiTIMED_Dataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        self.cell_types = dataframe.loc[:, 'Tumor':'Neu'] # inputs/features
        self.death = dataframe['survival_time'] # ouput
        self.status = dataframe['vital_status'] # output
        
    def __len__(self):
        return len(self.df.index)
        
    def __getitem__(self, idx):
        cell_types = torch.tensor(self.cell_types.iloc[idx]).to(device)
        death = torch.tensor(int(float(self.death.iloc[idx]))).to(device)
        status = torch.tensor(self.status.iloc[idx]).to(device)
        return cell_types, death, status


# In[5]:


TCGA_HiTIMED_Data = TCGA_HiTIMED_Dataset(tcga_hitimed)


# In[6]:


indices = torch.randperm(len(TCGA_HiTIMED_Data)).tolist()
dataset_train = torch.utils.data.Subset(TCGA_HiTIMED_Data, indices[:-96])
dataset_val = torch.utils.data.Subset(TCGA_HiTIMED_Data, indices[-96:-48])
dataset_test = torch.utils.data.Subset(TCGA_HiTIMED_Data, indices[-48:])


# In[7]:


batch_size = 16


# In[8]:


data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
data_loader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)


# In[9]:


class HiTIMED_MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(HiTIMED_MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)
        return x


# In[10]:


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


input_size = 17
hidden_sizes = [64, 64, 64, 64]  # [64, 32, 16]  # [20, 20, 20, 20]
output_size = 1
model = HiTIMED_MLP(input_size, hidden_sizes, output_size).to(device)

loss_fn = cox_loss

learning_rate = 0.00007
num_epochs = 60  # 1000
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.1)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

# model


# In[31]:


# Set the start method for multiprocessing to 'spawn'
# mp.set_start_method('spawn')
# print(mp.get_start_method(allow_none =True))

# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(data_loader_train), epochs=num_epochs)

model.train()
ftrain_losses = []
fval_losses = []
ftrain_c_indices = []
fval_c_indices = []
for epoch in range(num_epochs):
    train_losses = []
    val_losses = []
    train_c_indices = []
    val_c_indices = []
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
            # y_status = ~y_status ########################
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
        val_losses.append(loss.item())
        
        with torch.no_grad():
            x, y_time, y_status = input_data
            # y_status = ~y_status ########################
            x = x.float().to(device)
            y_time = y_time.float().cpu().numpy()
            y_status = y_status.float().cpu().numpy()
            output = model(x)
            c_index = concordance_index(y_time, -output.cpu(), y_status)
            val_c_indices.append(c_index)
            
    ftrain_losses.append(sum(train_losses)/len(train_losses))
    fval_losses.append(sum(val_losses)/len(val_losses))
    
    ftrain_c_indices.append(sum(train_c_indices)/len(train_c_indices))
    fval_c_indices.append(sum(val_c_indices)/len(val_c_indices))
    
    scheduler.step()
    
    print('Epoch %d | Loss %6.2f' % (epoch, sum(train_losses)/len(train_losses)))


# In[32]:


import matplotlib.pyplot as plt


# In[33]:


plt.plot(ftrain_losses, label='Training Loss')
plt.plot(fval_losses, label='Validation Loss')
# Add labels, title, and legend
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# In[34]:


plt.plot(ftrain_c_indices, label='Train C Indices')
plt.plot(fval_c_indices, label='Validation C Indices')
plt.xlabel('Epochs')
plt.ylabel('C Index')
plt.title('Training and Validation C Index')
plt.legend()
plt.show()

print(max(fval_c_indices))


# In[ ]:


from lifelines.utils import concordance_index

model.eval()
c_indices = []
for batch_num, input_data in enumerate(data_loader_test):
    with torch.no_grad():
        x, y_time, y_status = input_data
        x = x.float().to(device)
        y_time = y_time.float().cpu().numpy()
        y_status = y_status.float().cpu().numpy()
        output = model(x)
        c_index = concordance_index(y_time, -output.cpu(), y_status)
        c_indices.append(c_index)
        print(y_time)
        print(y_status) 

print(sum(c_indices) / len(c_indices))


# In[ ]:


# DO NOT RUN THE CELL BELOW UNLESS THE MODEL IS GOOD


# In[ ]:


torch.save(model.state_dict(), '/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Interns_2023/user/TristanD/HiTIMED/HiTIMED_MODEL_3.pt')


# In[ ]:




