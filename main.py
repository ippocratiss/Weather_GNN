from Data import (Get_UK_Stations, Fetch_Station_Data,Gather_Data,
                  Normalise,Connect_Graph,Build_Graph) 

from datetime import datetime
from utils import Visualise_Graph, train, eval, Epoch_Plots
from model import Weather_GNN
from torch_geometric.loader import DataLoader
import torch

#Collect and prepare data. 
start = datetime(2023,1,1)
end = datetime(2025,1,1)
features = ['temp','pres','rhum']
pred_window = 6
stations_df = Get_UK_Stations()
Weather_data = Gather_Data(stations_df,start,end,features)
print(f'Collectd dada from {len(Weather_data.keys())} stations...\n')
Weather_data = Normalise(Weather_data,features) 
SIDs = list(Weather_data.keys())
edge_index = Connect_Graph(stations_df,SIDs,3)
Graph = Build_Graph(Weather_data,edge_index,pred_window) 
#Visualise_Graph(Weather_data,stations_df,edge_index,SIDs)

#use GPU, if you have access to nvidia GPU use cuda, for macos use mps-
  # if neither is available use cpu. 
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}\n")

#Define the model
time_step = 7
n_feat = 3
model = Weather_GNN(n_feat,
                    64,128,
                    n_feat,time_step).to(device)

optimizer = torch.optim.NAdam(model.parameters(), lr=2*10**-4)
loss_fn = torch.nn.MSELoss()
#define Training, Validation and test sets
train_loader = DataLoader(Graph[:11500], batch_size=14, shuffle=True)
val_loader = DataLoader(Graph[12000:13000],batch_size = 14)
test_loader = DataLoader(Graph[17000:], batch_size=1)
#reduce LR if needed
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.8,threshold=10**-3)
#Training Loop 
epochs = 100
best_val_loss = float('inf')
tLoss = []
vLoss = []
r2vals = []
l_r = []
tstep = 6
N_features = 3
for epoch in range(epochs):
    train_loss = train(train_loader, model,
                             optimizer, loss_fn,N_features,
                             tstep,device)
    r2,val_loss = eval(val_loader, model,N_features,
                       tstep,loss_fn,device)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
    scheduler.step(val_loss)
    tLoss.append(train_loss)
    vLoss.append(val_loss)
    r2vals.append(r2)
    l_r.append(scheduler.get_last_lr()[0])
    print(f'EPOCH {epoch+1}:')
    print(f'    Training Loss = {train_loss:.5f}\n    Validation Loss = {val_loss:.5f}\n    r^2 Score = {r2:.5f}')
    print(f'    LR = {scheduler.get_last_lr()[0]:.4e}')

Epoch_Plots(tLoss,vLoss,r2vals,l_r)

