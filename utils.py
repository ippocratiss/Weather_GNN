import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
import torch 
from sklearn.metrics import r2_score
import numpy as np 
def Visualise_Graph(Weather_data,sts_df,edge_index,Station_ids):
    station_coords = {
    row['id']: (row['latitude'], row['longitude'])
    for _, row in sts_df.iterrows()
    if row['id'] in Weather_data
    }

    lats, lons = zip(*station_coords.values())

    # Set up map
    fig = plt.figure(figsize=(4.5,8.5))
    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_extent([-9, 2, 48.5, 62], crs=ccrs.PlateCarree())  # UK bounds

    # map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN)
    ax.gridlines(draw_labels=True)
    ax.set_position([0.12, 0.05, 0.7, 0.85])
    # Plot stations

    for I,J in edge_index.t().tolist():
        id1 = Station_ids[int(I)]
        id2 = Station_ids[int(J)]
        if id1 in station_coords and id2 in station_coords:
            lat1, lon1 = station_coords[id1]
            lat2, lon2 = station_coords[id2]
            ax.plot([lon1, lon2], [lat1, lat2],
                    transform=ccrs.PlateCarree(), color='blue', linewidth=2, alpha=1)
    ax.scatter(lons, lats, color='black', s=150, transform=ccrs.PlateCarree(), label="Stations")
    plt.title("UK Weather Stations")
    plt.legend()
    plt.show()
#Training function 
def train(loader, model, optimizer, loss_fn,nfeat,pstep,device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        time_steps = batch.x.size(1) // nfeat
        num_nodes = batch.num_nodes
        #print('time_steps=',time_steps)
        optimizer.zero_grad()
        x_series = batch.x.reshape(num_nodes, time_steps, nfeat)
        loss = 0
        targets = batch.y.size(1) // nfeat
        for stp in range(targets):
            x = x_series.reshape(num_nodes,-1)
            out = model(x, batch.edge_index)
            target = batch.y[:, stp * nfeat:(stp + 1) * nfeat]
            #Put Prediction into the sequence
            #print(f"out: {out.shape}, target: {target.shape}, y: {batch.y.shape}, step={stp}")
            loss += loss_fn(out, target)
            pred_reshaped = out.unsqueeze(1)
            x_series = torch.cat([x_series[:, 1:]
                                  ,pred_reshaped], dim=1)
        loss.backward(retain_graph = False)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
#Validation / Test function

def eval(loader, model,nfeat,psteps,loss_fn, device):
    model.eval()
    preds = []
    targets = []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_series = batch.x
            num_nodes = batch.num_nodes
            predictions = []
            inputs = x_series.view(num_nodes, -1, nfeat)
            for _ in range(psteps):
                x = inputs.view(num_nodes,-1)
                out = model(x,batch.edge_index)
                predictions.append(out)
                out_unsq = out.unsqueeze(1)
                inputs = torch.cat([inputs[:, 1:], out_unsq], dim=1)
            pred_series = torch.stack(predictions,1)
            #print("batch.y.shape =", batch.y.shape)
            #print('batch.y.shape = ', batch.y.shape)
            target_series = batch.y.view(num_nodes,psteps,nfeat).to(device)
            loss = loss_fn(pred_series, target_series)
            total_loss += loss.item()
            preds.append(pred_series.detach().cpu())
            targets.append(target_series.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    r2 = r2_score(targets.reshape(-1, targets.shape[-1]),
                  preds.reshape(-1, preds.shape[-1]))
    #rmse = np.sqrt(mean_squared_error(targets,preds))
    return (r2,total_loss/len(loader))

def predict(graph, model,nfeat,psteps, device):
    model.eval()
    preds = []
    targets = []
    total_loss = 0
    graph = [graph]
    with torch.no_grad():
        for batch in graph:
            batch = batch.to(device)
            x_series = batch.x
            num_nodes = batch.num_nodes
            predictions = []
            inputs = x_series.view(num_nodes, -1, nfeat)
            for _ in range(psteps):
                x = inputs.view(num_nodes,-1)
                out = model(x,batch.edge_index)
                predictions.append(out)
                out_unsq = out.unsqueeze(1)
                inputs = torch.cat([inputs[:, 1:], out_unsq], dim=1)
            pred_series = torch.stack(predictions,1)
            #print("batch.y.shape =", batch.y.shape)
            #print('batch.y.shape = ', batch.y.shape)
            target_series = batch.y.view(num_nodes,psteps,nfeat).to(device)
            #loss = loss_fn(pred_series, target_series)
            #total_loss += loss.item()
            preds.append(pred_series.detach().cpu())
            targets.append(target_series.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    r2 = r2_score(targets.reshape(-1, targets.shape[-1]),
                  preds.reshape(-1, preds.shape[-1]))
    #rmse = np.sqrt(mean_squared_error(targets,preds))
    return (r2,preds,targets)

def Epoch_Plots(t_loss,v_loss,r2,LR):
    e = np.arange(len(r2))
    fig,ax = plt.subplots(2,2)
    ax[0,0].plot(e,t_loss)
    ax[0,0].set_title('Training Loss')
    ax[0,1].plot(e,v_loss)
    ax[0,1].set_title('Validation Loss')
    ax[1,0].plot(e,r2)
    ax[1,0].set_title('r^2 Score')
    ax[1,1].plot(e,LR)
    ax[1,1].set_title('Learning Rate')
    plt.show()