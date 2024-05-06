import pandas as pd
pd.options.display.max_rows = 10
import os
import datetime
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import pathlib
import pickle
import numpy as np
from tqdm import tqdm

def read_raw_pickle() -> dict:
    pickle_path = os.path.dirname(os.path.realpath(__file__)) + '/train_data.pkl'  # Update file path
    dict_of_df = pd.read_pickle(pickle_path)
    return dict_of_df

def get_time_code(dt):
    hour = dt.hour
    if hour < 8:
        return 0.0
    elif hour < 16:
        return 0.5
    else:
        return 1.0

def area_block(lon, lat):
    if (lon < 113.8) or (lat < 22.45) or (lon > 114.3) or (lat > 22.85): return -1
    lon_val = int((lon - 113.8)/0.1)
    lat_val = int((lat - 22.45)/0.1)
    return (lon_val + lat_val*6)/30

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def calculate_distance_and_speed(df : pd.DataFrame):
    # Calculate distance using haversine formula
    df['longitude_shift'] = df['longitude'].shift().fillna(0)
    df['latitude_shift'] = df['latitude'].shift().fillna(0)
    df['distance'] = df.apply(lambda row: haversine(row['longitude'], row['latitude'], row['longitude_shift'], row['latitude_shift']), axis=1)
    
    # Calculate speed in km/h then divide by 60 
    df['speed'] = df['distance'] / df['time_diff'] * 3600 / 60
    
    return df

def feature_engineer(list_array : 'list[np.array]') -> 'list[pd.DataFrame]':
    list_of_columns = ['plate','longitude','latitude','second_since_midnight','status','time']
    list_of_df = []
    for plate_list in list_array:
        plate_df = pd.DataFrame(data=plate_list,columns=list_of_columns)
        
        plate_df['time'] = plate_df.apply(lambda row: datetime.strptime(row['time'], '%Y-%m-%d %H:%M:%S'), axis=1)
        # plate_df['time_interval'] = plate_df['time'].apply(lambda x: int(x.timestamp())) - plate_df['time'].shift(1).fillna(pd.Timestamp.min).apply(lambda x: int(x.timestamp()))
        plate_df['time_diff'] = plate_df['time'].diff().dt.total_seconds()
        plate_df['time_code'] = plate_df['time'].apply(lambda x: get_time_code(x))
        plate_df['is_workday'] = plate_df['time'].apply(lambda x: 1 if x.weekday() in range(0, 5) else 0)
        plate_df['area_block'] = plate_df.apply(lambda row: area_block(row['longitude'], row['latitude']),axis=1)
        
        plate_df = calculate_distance_and_speed(plate_df)
        plate_df = plate_df.tail(-1)
        # scaler = StandardScaler()
        
        plate_df = plate_df.replace([np.inf, -np.inf], np.nan)
        plate_df = plate_df.dropna()
        
        plate_df.drop('plate',axis=1)
        
        list_of_df.append(plate_df)
    return list_of_df

def slice_dataframe(list_of_df, sequence_length = 100):
    days_of_traj = []
    max_length = 512  # set the maximum sequence length to 512
    for plate_df in list_of_df:
        result = []
        values = plate_df[['distance','speed', 'status', 'time_code','is_workday','area_block']].values.tolist()
        
        # Processing to ensure each sublist has a length of 100
        while len(values) > 0:
            if len(values) >= sequence_length:
                days_of_traj.append(values[:sequence_length])
                values = values[sequence_length:]
            else:
                padding = [[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]] * (sequence_length - len(values))
                days_of_traj.append(values + padding)
                break

    return days_of_traj

if __name__ == '__main__': 
    dict_of_df = read_raw_pickle()
    all_trajectories = {}
    pickle_dir = os.path.dirname(os.path.realpath(__file__))
    for i in tqdm(dict_of_df.keys()):  # Loop over actual keys
        cur_list_of_traj = dict_of_df[i]
        # The rest of your processing...

        try:
            engineered_traj = feature_engineer(cur_list_of_traj)
            # Specify the desired sequence length here
            sliced_traj = slice_dataframe(engineered_traj, sequence_length=100)
            all_trajectories[i] = sliced_traj
        except Exception as e:
            print(f"Error processing trajectory {i}: {e}")
            continue

    # Save the dictionary of all processed trajectories to a single pickle file
    pickle_file_path = os.path.join(pickle_dir, 'train400_feature_all.pkl')  # Update file name
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(all_trajectories, f) 
