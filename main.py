from Data import (Get_UK_Stations,Gather_Data,
                  Connect_Graph,Build_Graph, Normalise_fit_on_train) 

from datetime import datetime
from utils import Visualise_Graph, train, eval, Epoch_Plots
from model import Weather_GNN
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import joblib 
#Collect and prepare data. 
start = datetime(2020,1,1)
end = datetime(2025,4,1)
features = ['temp','pres']
pred_window = 12
lookback = 48
trainFrac = 0.8
stations_df = Get_UK_Stations()
Weather_data = Gather_Data(stations_df,start,end,features)
print(f'Collected data from {len(Weather_data.keys())} stations...\n')

# split data into training and validation sets. 
# Normalise data on training set
nHr = len(next(iter(Weather_data.values())))
npos = nHr - (pred_window + lookback)
ntrain = int(npos * trainFrac)
last_train_T = lookback + ntrain - 1
train_end_idx = last_train_T + pred_window
#scale data on training set. 
Weather_data, scalers = Normalise_fit_on_train(Weather_data, features, train_end_idx)

#connect graph nodes 
SIDs = list(Weather_data.keys())
#Each node connected to 7 nearest.
edge_index,edge_attr = Connect_Graph(stations_df,SIDs,7)
Graph = Build_Graph(Weather_data,edge_index,edge_attr,pred_window,lookback) 

#Uncomment below to visualize GNN structure. 
#Visualise_Graph(Weather_data,stations_df,edge_index,SIDs)

#Save model preprocessing and metadata. 
preproc = {
    "scalers": scalers,               
    "features": features,              
    "lookback": lookback,
    "pred_window": pred_window,
    "time_step": lookback + 1,
    "SIDs": SIDs,                     
    "nearest": 7,                      
    "edge_index": edge_index.cpu(),
    "edge_attr": edge_attr.cpu(),
}
joblib.dump(preproc,"preproc.joblib")

train_graphs = Graph[:ntrain]
val_graphs   = Graph[ntrain:int(0.98 * len(Graph))] 


#use GPU, if you have access to nvidia GPU use cuda, for macos use mps-
  # if neither is available use cpu. 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

#Define the model
time_step = lookback + 1 
n_feat = len(features)
print(f'Length of data: {len(next(iter(Weather_data.values())))} Hrs \n')
model = Weather_GNN(n_feat,128,
                    len(features)*pred_window,time_step).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2*10**-4,weight_decay=0.00001)
loss_fn = torch.nn.MSELoss()
#define traiing and validation sets. 
tot = len(Graph)
train_loader = DataLoader(train_graphs, batch_size=10, shuffle=True)
val_loader = DataLoader(val_graphs,batch_size = 10)

#reduce LR if needed
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.7,threshold=10**-4)
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

#Training loop 
for epoch in range(epochs):
    #Early stopping if no improvement after 30 epochs. 
    if earlystop > 30:
        print(f'Training stopped at Epoch {epoch+1}')
        break
    #Get training/validation loss and r^2 score and update weights. 
    train_loss = train(train_loader, model,
                             optimizer, loss_fn,N_features,
                             tstep,device)
    r2,val_loss = eval(val_loader, model,N_features,
                       tstep,loss_fn,device)
    #save best model. 
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        print('\n Best Model Saved')
        earlystop = 0 
    else:
        earlystop += 1
    #update LR scheduler 
    scheduler.step(val_loss)
    tLoss.append(train_loss)
    vLoss.append(val_loss)
    r2vals.append(r2)
    l_r.append(scheduler.get_last_lr()[0])
    print(f'EPOCH {epoch+1}:')
    print(f'    Training Loss = {train_loss:.5f}\n    Validation Loss = {val_loss:.5f}\n    r^2 Score = {r2:.5f}')
    print(f'    LR = {scheduler.get_last_lr()[0]:.4e}')
#show epoch plots at end of training 
Epoch_Plots(tLoss,vLoss,r2vals,l_r)

