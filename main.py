import torch.optim.radam
from Data import (Get_UK_Stations, Fetch_Station_Data,Gather_Data,
                  Normalise,Connect_Graph,Build_Graph) 

from datetime import datetime
from utils import Visualise_Graph, train, eval, Epoch_Plots
from model import Weather_GNN
from torch_geometric.loader import DataLoader
import torch
import numpy as np
#Collect and prepare data. 
start = datetime(2022,7,1)
end = datetime(2025,3,1)
features = ['temp','pres']
pred_window = 12
lookback = 48
stations_df = Get_UK_Stations()
Weather_data = Gather_Data(stations_df,start,end,features)
print(f'Collectd dada from {len(Weather_data.keys())} stations...\n')
Weather_data ,_ = Normalise(Weather_data,features) 
SIDs = list(Weather_data.keys())
edge_index,edge_attr = Connect_Graph(stations_df,SIDs,6)
Graph = Build_Graph(Weather_data,edge_index,edge_attr,pred_window,lookback) 
#Visualise_Graph(Weather_data,stations_df,edge_index,SIDs)

#use GPU, if you have access to nvidia GPU use cuda, for macos use mps-
  # if neither is available use cpu. 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

#Define the model
time_step = 49
n_feat = len(features)
print(f'Length of data: {len(next(iter(Weather_data.values())))} Hrs \n')
model = Weather_GNN(n_feat,256,
                    len(features)*pred_window,time_step).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1*10**-4,weight_decay=0.1)
loss_fn = torch.nn.MSELoss()
#define Training, Validation and test sets
tot = len(Graph)
train_loader = DataLoader(Graph[:int(0.8*tot)], batch_size=10, shuffle=True)
val_loader = DataLoader(Graph[int(0.8*tot):int(0.98*tot)],batch_size = 10)

#reduce LR if needed
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=6, factor=0.7,threshold=10**-4)
#Training Loop 
epochs = 300
best_val_loss = float('inf')
tLoss = []
vLoss = []
r2vals = []
l_r = []
tstep = 12
N_features = n_feat
earlystop = 0 
print('number of features:',n_feat)
for epoch in range(epochs):
    if earlystop > 25:
        print(f'Training stopped at Epoch {epoch+1}')
        break
    train_loss = train(train_loader, model,
                             optimizer, loss_fn,N_features,
                             tstep,device)
    r2,val_loss = eval(val_loader, model,N_features,
                       tstep,loss_fn,device)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        print('\n Best Model Saved')
        earlystop = 0 
    scheduler.step(val_loss)
    tLoss.append(train_loss)
    vLoss.append(val_loss)
    r2vals.append(r2)
    l_r.append(scheduler.get_last_lr()[0])
    earlystop +=1
    print(f'EPOCH {epoch+1}:')
    print(f'    Training Loss = {train_loss:.5f}\n    Validation Loss = {val_loss:.5f}\n    r^2 Score = {r2:.5f}')
    print(f'    LR = {scheduler.get_last_lr()[0]:.4e}')
Epoch_Plots(tLoss,vLoss,r2vals,l_r)

