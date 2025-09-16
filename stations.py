import joblib 
from Data import Get_UK_Stations, find_station_names
preproc = joblib.load("preproc.joblib")
stations_df = Get_UK_Stations()
trained_sids = preproc["SIDs"]
names = find_station_names(trained_sids,stations_df)
for i in range(0,len(names),3):
    stations = names[i:i+3]
    for n in range(len(stations)):
        print(f'{i + n + 1}.{stations[n]}   ') 
    
