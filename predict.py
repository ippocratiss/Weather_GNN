from model import Weather_GNN
import torch 
from Data import (Get_UK_Stations, Fetch_Station_Data,Gather_Data,
                  Normalise,Connect_Graph,Build_Graph,denormalise,
                  find_station_id) 
from datetime import datetime
from torch_geometric.loader import DataLoader
from utils import predict
import matplotlib.pyplot as plt 
import numpy as np 
start = datetime(2023,1,1)
end = datetime(2025,1,1)
features = ['temp','pres']
pred_window = 6
stations_df = Get_UK_Stations()
Weather_data = Gather_Data(stations_df,start,end,features)
print(f'Collectd dada from {len(Weather_data.keys())} stations...\n')
Weather_data,scalers = Normalise(Weather_data,features) 
SIDs = list(Weather_data.keys())
edge_index = Connect_Graph(stations_df,SIDs,3)
Graph = Build_Graph(Weather_data,edge_index,pred_window) 

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
time_step = 7
n_feat = len(features)
model = Weather_GNN(n_feat,128,128, 2, time_step).to(device)
model.load_state_dict(torch.load('best_model.pt'))
test_graph = Graph[-2].to(device)
test_r2,pred,targ = predict(test_graph,model, n_feat,pred_window,device)

station_names = ['london','bournemouth','glasgow']
ids = []
for n in station_names:
    id,name = find_station_id(stations_df,n,SIDs)
    ids.append([id,name])
station_index = [SIDs.index(sid[0]) for sid in ids]
idx = 0 
thrs = np.arange(0,7)
T_input = test_graph.x.view(test_graph.num_nodes, time_step, n_feat)[:, -1].cpu().numpy()
for i in station_index:
    sid = SIDs[i]
    scaler = scalers[sid] 
    pred_i = pred[i]
    target_i = targ[i] 
    Tval = T_input[i].reshape(1,-1)
    pred_dnorm = denormalise(pred_i,scaler)
    targ_dnorm = denormalise(target_i,scaler)
    T_dnorm = denormalise(Tval,scaler)
    fig, ax = plt.subplots(1,2)

    pred_plot = np.vstack([T_dnorm,pred_dnorm])
    targ_plot = np.vstack([T_dnorm,targ_dnorm])
    for j, feat in enumerate(features):
        ax[j].plot(thrs, targ_plot[:, j], label=f'Target {feat}')
        ax[j].plot(thrs, pred_plot[:, j], '--', label=f'Predicted {feat}')
        ax[j].set_title(f'Station: {ids[idx][1]} — Forecast for Next {pred_window} Hours')
        ax[j].set_xlabel('Hour Ahead')
        ax[j].set_ylabel('Value')
        ax[j].set_xticks(thrs)
        joined = list(targ_dnorm[:,j])+list(pred_dnorm[:,j])
        ymin = min(joined) - 3
        ymax = max(joined ) + 3
        ax[j].set_ylim(ymin,ymax)
        ax[j].legend()
    idx += 1
    plt.tight_layout()
    plt.show() 
