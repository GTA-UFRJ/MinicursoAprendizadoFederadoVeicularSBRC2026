import torch

import pandas as pd

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

def load_tp(client_id:int=1, 
            data_path:str="data/processed/", 
            speed:int=0, 
            data_file:str="0.csv"):
    
    client_id = 1
    df = pd.read_csv(f"{data_path}/{data_file}")
    dt = df[df['Node ID'] == client_id].reset_index()
    tpu = dt[['Throughput DL']].values.astype('float32')
    tpd = dt[['Throughput UL']].values.astype('float32')
    
    return tpu, tpd

