import torch
from torch_geometric.data import Data
from meteostat import Stations, Hourly
from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import numpy as np 
def Get_UK_Stations():
    #Get UK stations from meteostat api
    sts = Stations().fetch()
    return sts[sts['country'] == 'GB'].reset_index()

def Fetch_Station_Data(SID,start,end,features,length):
    try:
        #Ensure that each dataset for each ID has complete data
        data = Hourly(SID,start,end).fetch().astype(float)
        if len(data) != length:
            return None, f'Station: {SID}, has incomplete data'
        if not all(f in data.columns for f in features):
            return None, f"Station: {SID} missing features: {data.columns}"
        if data[features].isnull().values.any():
            return None, f'Station {SID} has missing data'
        return data[features].copy().astype(float), None
    except Exception as e:
        return None,f"Could not fetch data for {SID}: {e}"
    
def Gather_Data(sts_df, start,end,feats):
    length = len(Hourly('03761',start,end).fetch())
    Weather_Data = {}
    bad_ids = []
    #collect usable data and store bad ID's 
    for _,row in sts_df.iterrows():
        SID = row['id']
        data,error = Fetch_Station_Data(SID,start,end,feats,length)
        if data is not None:
            Weather_Data[SID] = data
        else:
            bad_ids.append((SID,error))
    print(f'{len(bad_ids)} stations could not be used... \n')
    return Weather_Data 

def Normalise(Weather_Data,feats):
    #Normalise Data 
    scalers = {}
    for ID in Weather_Data:
        scaler = MinMaxScaler()
        Weather_Data[ID][feats] = scaler.fit_transform(Weather_Data[ID][feats])
        scalers[ID] = scaler
    return Weather_Data,scalers 

def Connect_Graph(sts_df,SID,nearest):
    #connect each node to its N nearest edges 
    id_idx = {ID: idx for idx,ID in enumerate(SID)}
    coordinates = {row['id']: (row['latitude'],row['longitude'])
                   for _, row in sts_df.iterrows()
                   if row['id'] in SID
                   } 
    edge_index = []
    #edge_attr = [] 
    for ID in SID:
        dists = []
        lat1,lon1 = coordinates[ID]
        if lat1 < 0:
            continue 
        for ID2 in SID:
            if ID2 != ID:
                lat2,lon2 = coordinates[ID2]
                D = geodesic((lat1,lon1),(lat2,lon2)).km
                dists.append((D,ID2))
        dists = sorted(dists)[:nearest]
        for ID3 in dists:
            i,j = int(id_idx[ID]),int(id_idx[ID3[1]])
            edge_index.append([i,j])
            edge_index.append([j,i])
            #edge_attr.append([ID3[0]])            
            #edge_attr.append([ID3[0]])
    edge_index = np.array(edge_index,dtype=np.int64)
    edge_index = torch.tensor(edge_index,dtype=torch.long).t().contiguous() 

    #The below lines can be uncommented in edge_attr is wanted...  
    #scaler = MinMaxScaler()
    #edge_attr = np.array(edge_attr,dtype=np.float32)
    #edge_attr = scaler.fit_transform(edge_attr)
    #edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return edge_index

def Build_Graph(Weather_Data,edge_index,p_window):
    #input node features as time sequence for GRU 
    n_hrs = len(next(iter(Weather_Data.values())))
    station_ids = list(Weather_Data.keys())
    graph_data_list = []
    for T in range(p_window, n_hrs-6):
        Xnode = []
        Ynode = []
        for SID in station_ids:
            df = Weather_Data[SID]
            Window = df.iloc[T - p_window:T+1].values
            Target = df.iloc[T+1:(T+1)+p_window].values
            Xnode.append(Window.flatten())
            Ynode.append(Target.flatten())
        Xnode = np.array(Xnode)
        Ynode = np.array(Ynode)
        X = torch.tensor(Xnode, dtype=torch.float)
        Y = torch.tensor(Ynode, dtype=torch.float)
        graph = Data(x=X, edge_index=edge_index, y=Y.clone())
        graph_data_list.append(graph)

        #the shape of each graph should be as follows...
        #graph.x = (num_nodes,(p_window + 1)*n_features)
        #graph.edges = (2,num_edges)
        #graph.y = (num_nodes,p_window*n_features)
        
    return graph_data_list 

def denormalise(data,scaler):
    return scaler.inverse_transform(data) 

def find_station_id(sts_df,name,SIDs):
    station = sts_df[sts_df['name'].str.contains(name,case=False,na=False)]
    l = len(station)
    if not station.empty:
        for i in range(l):
            if station.iloc[i]['id'] in SIDs:
                return [station.iloc[i]['id'],station.iloc[i]['name']]
            
    else:
        print(f'No matches for{name}')
        return None 
