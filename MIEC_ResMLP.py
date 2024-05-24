import pandas as pd
import torch
import numpy as np
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
import math
import time
from tqdm import tqdm
import random
import copy

class split_dataset():
    def __init__(self, filepath, split_prob):
        super().__init__()
        self.filepath = filepath
        self.data = pd.read_csv(filepath,header=None)
        self.data  = self.data.sample(frac =1).reset_index(drop=True)
        self.num_group = self.data.loc[:,:1].groupby(self.data.columns[1]).size().sort_values().to_dict() 
        self.split_prob = split_prob
    def split_data(self, train_path, test_path):
        train_data = self.data.groupby(self.data.columns[1]).apply(self.typicalsamling).copy().reset_index(drop=True)
        test_data = self.data.append(train_data).drop_duplicates(keep=False).copy().reset_index(drop=True)
        train_data.to_csv(train_path, index=False, header=None)
        test_data.to_csv(test_path, index=False, header=None)
        return train_data,test_data      
    def typicalsamling(self, group):
        name = group.name
        n = self.num_group[name]
        #if n > 1:
        return group.sample(n=int(n*self.split_prob))
        #else:
        #    print(name) 

class mydataset():
    def __init__(self,data,x_idx,y_idx):
        super().__init__()
        #self.filepath = filepath
        self.data = data
        self.x = x_idx
        self.y = y_idx
        #self.aug_prob = aug_prob 
        #self.block_size = block_size
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x=torch.tensor(np.array(self.data.iloc[idx,self.x],dtype=np.float32))*2
        y=torch.tensor(np.array(self.data.iloc[idx,self.y],dtype=np.float32))
        all = torch.sum(y,dim=0)
        y = y.div(all)
        return x,y
def split_data():    
    path="./MIEC_ResMLP/data.csv"
    train_path="./MIEC_ResMLP/train_data.csv"
    test_path="./MIEC_ResMLP/test_data.csv"
    prob=0.8
    dataset = split_dataset(path,prob)
    train_dataset,test_dataset = dataset.split_data(train_path,test_path)
    x_idx=[i for i in range(2,339)]
    y_idx=[i for i in range(339,347)]
    train_data = mydataset(train_dataset,x_idx,y_idx)
    test_data = mydataset(test_dataset,x_idx,y_idx)
    return train_data,test_data

class MLP (nn.Module):
    def __init__(self,num_i,num_o,num_h,num_d,scale):
        super().__init__()
        self.ln1 = nn.Linear(num_i,num_h)
        self.re1 = nn.ReLU()
        self.ln2 = nn.Linear(num_h,num_h)
        self.re2 =  nn.ReLU()   
        self.drop = nn.Dropout(num_d)
        self.ln3 = nn.Linear(num_h,num_o)
    def forward(self,x):
        y = self.ln1(x) 
        y = self.re1(y)
        y = y + self.ln2(y)
        y = self.re2(y)
        y = self.drop(y)
        y = self.ln3(y)
        return y


def train_model(model,dataset,epoch,batch_size=32,lr=0.1,is_train=True):
    cost = nn.CrossEntropyLoss()
    data = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    loop = tqdm(enumerate(data), total=len(data)) if is_train else enumerate(data)
    batch_losses=[]
    for it,(x,y) in loop:
        if is_train:
                model.train()
                outputs = model(x)
                optimizer.zero_grad()
                loss = cost(outputs.squeeze(),y)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
                loop.set_description(f"epoch {epoch+1}. iter {it}. train loss {loss.item():.5f}. lr {lr:e}")
                  
        else:
                model.eval()
                outputs = model(x)
                loss = cost(outputs.squeeze(),y)
                loss = loss.mean()
                batch_losses.append(loss.item())  
    epoch_loss = np.mean(batch_losses)
    return epoch_loss


def training(train_data,test_data,epochs,num_i,num_o,num_h,num_d,scale,batch_size,lr):
    model=MLP(num_i,num_o,num_h,num_d,scale)
    train_losses=[]
    test_losses = []
    best_loss = None

#StepLR = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    for epoch in range(epochs):
        train_loss = train_model(model,train_data,epoch,batch_size,lr,is_train=True)
        test_loss = train_model(model,test_data,epoch,batch_size,lr,is_train=False)
    #StepLR.step()
    #lr = StepLR.get_last_lr()[0]
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if best_loss is not None:
            if test_loss < best_loss:
                best_loss = test_loss
            #print(f'save at:{best_loss}')
                torch.save(model.state_dict(), './MIEC_ResMLP/mlp_out_best_.pt')
        else:
            best_loss=test_loss  
    return train_losses,test_losses

def eval_model(num_i,num_o,num_h,num_d,scale,test_data):
    model = MLP(num_i,num_o,num_h,num_d,scale)
    model.load_state_dict(torch.load("./MIEC_ResMLP/mlp_out_best_.pt"))
    model.eval()
    a = []
    b = []
    for i in range(test_data[:][1].shape[0]):
        y_all = test_data[i][1]
        y_predict = model(test_data[i][0])
        y_predict = F.softmax(y_predict)
        a.append(y_all)
        b.append(y_predict)
    a = torch.cat(a,dim=0).detach().numpy()
    b = torch.cat(b,dim=0).detach().numpy()

#y_predict = F.softmax(y_predict, dim=1)
#y_predict1 = y_predict.squeeze(1)
#y_all = y_all.reshape(1,-1)
#y_predict1 = y_predict1.reshape(1,-1)
#y_predict1 = y_predict1.detach().numpy()

    my_rho = np.corrcoef(a, b)
    my_rho
    return my_rho


