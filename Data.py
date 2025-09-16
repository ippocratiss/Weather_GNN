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

''' Args: 
        SID: A list of station ids.
        start: Start date of data collection.
        end: End date of data collection.
        features: list of weather features being collected.
        length: number of hours collected. 
    Returns:
        Data if available. 
        Error if data is missing.
    '''
def Fetch_Station_Data(SID,start,end,features,length):
    try:
        #This function is called from Gather_data()
        #Ensures that each dataset for each ID has complete data
        #If data is incomplete the station is not used. 
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


''' This function is called from main.py, it collects station ids
and the data from that station if available.
    Args:
        sts_df: Pandas DataFrame of UK stations.
        start: Start date of hourly data collection
        end: End date of hourly data collection
        feats: list of weather features to be collected.
    Returns:
        Weather_Data: A dictionary of station data'''
    
def Gather_Data(sts_df,start,end,feats):
    length = len(Hourly('03862',start,end).fetch())
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
    #Output number stations that had bad data
    print(f'{len(bad_ids)} stations could not be used... \n')
    return Weather_Data 

''' Normalize Data based on Training set.
    Args:
        Weater_Data: Dictionary of station data.
        feats: list of weather features to be collected.
        train_end_inx: index of last training data point.
    Returns: 
            Weather_Data: Scaled Dictionary of station data
            scalers: scalers used for normalization.'''
def Normalise(Weather_Data, feats, train_end_idx):
    #normalise data from training set and apply to all data.
    scalers = {}
    for sid, df in Weather_Data.items():
        scaler = MinMaxScaler()
        # fit on training slice only
        train_slice = df.iloc[:train_end_idx+1][feats].values  # +1 because iloc end is exclusive
        scaler.fit(train_slice)
        # transform full series
        df[feats] = scaler.transform(df[feats].values)
        scalers[sid] = scaler
    return Weather_Data, scalers

''' This function creates the edges between stations and saves
    scaled distances between stations as edge attributes.
    Args:
        sts_df: pandas dataframe of stations.
        SID: list of station Id's.
        nearest: number of nearest staions each node is connected to
    Returns:
        edge_index: tensor of edges between stations.
        edge_attribute: scaled distance between stations.'''
def Connect_Graph(sts_df,SID,nearest):
    #connect each node to its N nearest edges 
    id_idx = {ID: idx for idx,ID in enumerate(SID)}
    coordinates = {row['id']: (row['latitude'],row['longitude'])
                   for _, row in sts_df.iterrows()
                   if row['id'] in SID
                   } 
    edge_index = []
    edge_attr = [] 
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
            edge_attr.append([ID3[0]])            
            edge_attr.append([ID3[0]])
    edge_index = np.array(edge_index,dtype=np.int64)
    edge_index = torch.tensor(edge_index,dtype=torch.long).t().contiguous() 

    #The below lines can be uncommented in edge_attr is wanted...  
    scaler = MinMaxScaler()
    edge_attr = np.array(edge_attr,dtype=np.float32)
    edge_attr = scaler.fit_transform(edge_attr)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return edge_index,edge_attr
''' Build Graph cretes a list of tensors that contain station data, edges
    and edge attributes. This is the list that will be used in training and testing.
    Args: 
        Weather_Data: dictionary of station data.
        edge_index: tensor of edges between stations. 
        edge_attributes: scaled distance betweeen stations.
        p_window: length of time to be predicted.
        lockback: how many past hours are used for prediction.
    Returns:
        graph_data_list: list of tensors containing station data.'''
def Build_Graph(Weather_Data,edge_index,edge_attr,p_window,lookback):
    #input node features as time sequence for GRU 
    n_hrs = len(next(iter(Weather_Data.values())))
    station_ids = list(Weather_Data.keys())
    graph_data_list = []
    for T in range(lookback, n_hrs-p_window):
        Xnode = []
        Ynode = []
        for SID in station_ids:
            df = Weather_Data[SID]
            Window = df.iloc[T - lookback:T+1].values
            Target = df.iloc[T+1:(T+1)+p_window].values
            Xnode.append(Window.flatten())
            Ynode.append(Target.flatten())
        Xnode = np.array(Xnode)
        Ynode = np.array(Ynode)
        X = torch.tensor(Xnode, dtype=torch.float)
        Y = torch.tensor(Ynode, dtype=torch.float)
        graph = Data(x=X, edge_index=edge_index,edge_attr=edge_attr, y=Y.clone())
        graph_data_list.append(graph)

        #the shape of each graph should be as follows...
        #graph.x = (num_nodes,(p_window + 1)*n_features)
        #graph.edges = (2,num_edges)
        #graph.y = (num_nodes,p_window*n_features)
        
    return graph_data_list 
''' Used in predict.py to unscale data and get real values 
    of features. 
    Args:
        data: torch tensor of predicted values.
        scaler: scalers created in normalize function.
    Returns:
        unscaled data'''
def denormalise(data,scaler):
    return scaler.inverse_transform(data) 

''' Used in predict.py to find ID's of stations.
    Args:  
        sts_df: pandas Dataframe of stations
        SIDs: list of station id's
    Returns:
            [station id, station name] '''
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

def find_station_names(SIDs,sts_df):
    names = []
    count = 1
    for id in SIDs:
        station = sts_df[sts_df['id'] == id]['name'].iloc[0]
        names.append(f'{station}')
        count += 1 
    return names
