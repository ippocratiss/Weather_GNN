import matplotlib
matplotlib.use('TKAgg')
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
        x_series = batch.x
        optimizer.zero_grad()
        num_nodes = batch.num_nodes
        x_series = x_series.reshape(num_nodes,-1)
        out = model(x_series,batch.edge_index,batch.edge_attr)
        target = batch.y.view(num_nodes,pstep,nfeat).to(device)
        loss = loss_fn(out,target)
        loss.backward()
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
            num_nodes = batch.num_nodes
            x_series = batch.x.view(num_nodes,-1)
            target_series = batch.y.view(num_nodes,psteps,nfeat).to(device)
            out = model(x_series,batch.edge_index,batch.edge_attr)
            loss = loss_fn(out,target_series)
            total_loss += loss.item()
            preds.append(out.cpu())
            targets.append(target_series.cpu())
    preds = torch.cat(preds,dim=0).numpy()
    targs = torch.cat(targets,dim=0).numpy()
    r2 = r2_score(targs.reshape(-1, 2),
                  preds.reshape(-1,2))
    #rmse = np.sqrt(mean_squared_error(targets,preds))
    return (r2,total_loss/len(loader))

def predict(graph, model,nfeat,psteps, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        num_nodes = graph.num_nodes
        x_series = graph.x.view(num_nodes,-1)
        target_series = graph.y.view(num_nodes,psteps,nfeat).to(device)
        out = model(x_series,graph.edge_index,graph.edge_attr)
    preds = out.cpu().numpy()
    targs = target_series.cpu().numpy()
    #rmse = np.sqrt(mean_squared_error(targets,preds))
    return (preds,targs)
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

