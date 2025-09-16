from model import Weather_GNN
import torch 
from Data import (Get_UK_Stations,Gather_Data,
                  Connect_Graph,Build_Graph,denormalise,
                  find_station_id, find_station_names) 
from datetime import datetime
from torch_geometric.loader import DataLoader
from utils import predict
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import joblib
import os 

#load params from main.py 
preproc = joblib.load("preproc.joblib")
scalers = preproc["scalers"]
features = preproc["features"]
trained_SIDs = preproc["SIDs"]
nearest = preproc["nearest"]
time_step = preproc["time_step"]
pred_window = preproc["pred_window"]


#load graph for prediction 
start = datetime(2020,1,1)
end = datetime(2025,3,1)
lookback = 48
stations_df = Get_UK_Stations()
Weather_data = Gather_Data(stations_df,start,end,features)
print(f'Collected data from {len(Weather_data.keys())} stations...\n')
SIDs = list(Weather_data.keys())

for sid in Weather_data.keys():
    Weather_data[sid][features] = scalers[sid].transform(Weather_data[sid][features].values)


edge_index,edge_attr = Connect_Graph(stations_df,SIDs,nearest)
Graph = Build_Graph(Weather_data,edge_index,edge_attr,pred_window,lookback) 

# set up best model to make prediction. 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
time_step = 49
n_feat = len(features) 
model = Weather_GNN(n_feat,128, len(features)*pred_window, time_step).to(device)
model.load_state_dict(torch.load('best_model.pt'))
test_graph = Graph[-2].to(device)
pred,targ = predict(test_graph,model,n_feat,pred_window,device)

#Add any valid uk station name below to get prediction. 
station_names = ['london','bournemouth','glasgow']
#using all_stations will run predictions for all stations
#all_stations = find_station_names(SIDs,stations_df)

ids = []
outdr = os.path.join("Plots")
for n in station_names:
    id,name = find_station_id(stations_df,n,SIDs)
    ids.append([id,name])
station_index = [SIDs.index(sid[0]) for sid in ids]
idx = 0 
thrs = np.arange(0,13)
T_input = test_graph.x.view(test_graph.num_nodes, time_step, len(features))[:, -1].cpu().numpy()
#get predictions, denormalize values and plot. 
for i in station_index:
    sid = SIDs[i]
    scaler = scalers[sid] 
    pred_i = pred[i]
    target_i = targ[i] 
    Tval = T_input[i].reshape(1,-1)
    pred_dnorm = denormalise(pred_i[:, :len(features)],scaler)
    targ_dnorm = denormalise(target_i[:, :len(features)],scaler)
    T_dnorm = denormalise(Tval[:, :len(features)],scaler)
    fig, ax = plt.subplots(2,2)

    pred_plot = np.vstack([T_dnorm,pred_dnorm])
    targ_plot = np.vstack([T_dnorm,targ_dnorm])
    res = targ_plot - pred_plot
    for j, feat in enumerate(features):
        ax[0,j].plot(thrs, targ_plot[:, j], label=f'Target {feat}')
        ax[0,j].plot(thrs, pred_plot[:, j], '--', label=f'Predicted {feat}')
        ax[0,j].set_title(f'Station: {ids[idx][1]} — Forecast for Next {pred_window} Hours')
        ax[0,j].set_xlabel('Hour Ahead')
        ax[0,j].set_ylabel('Value')
        ax[0,j].set_xticks(thrs)
        joined = list(targ_dnorm[:,j])+list(pred_dnorm[:,j])
        ymin = min(joined) - 3
        ymax = max(joined ) + 3
        ax[0,j].set_ylim(ymin,ymax)
        ax[0,j].legend()

        ax[1,j].scatter(thrs,res[:,j])
        ax[1,j].axhline(0,ls='--')
        ax[1,j].set_title('Residual Error')
        ax[1,j].set_xlabel('Hour Ahead')
        ax[1,j].set_ylabel('Value')
    plt.tight_layout()
    #plt.show() 
    fig.savefig(os.path.join(outdr, f"{ids[idx][1].replace('/',' and ')}_prediction.png")) 
    idx += 1
