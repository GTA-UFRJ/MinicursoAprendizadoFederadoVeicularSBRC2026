# implement the training for lstm model to predict thorughputs
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.utils.data as data
import os

from utils.utils import load_config
from .lstm import LSTM
from .data import (
    create_dataset,
    load_tp,
)


def train(speed=0,
          base_station_range=1000, 
          PLOT=True):

    # train-test split for time series
    train_size = int(len(tpd) * 0.67)
    train, test = tpd[:train_size], tpd[train_size:]

    lookback = 5
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM()
    model.to(device)

    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), 
                             shuffle=True, 
                             batch_size=8)

    n_epochs = 40

    for epoch in range(n_epochs):
    
        model.train()

        for X_batch, y_batch in loader:
        
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
 
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        if epoch % 10 != 0:

            continue

        model.eval()

        with torch.no_grad():

            y_pred = model(X_train.to(device))
            train_rmse = torch.sqrt(loss_fn(y_pred, y_train.to(device)))
            y_pred = model(X_test.to(device))
            test_rmse = torch.sqrt(loss_fn(y_pred, y_test.to(device)))
        
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))


        with torch.no_grad():

            # shift train predictions for plotting
            train_plot = np.ones_like(tpd) * np.nan
            y_pred = model(X_train.to(device))
            y_pred = y_pred[:, -1, :]
            train_plot[lookback:train_size] = model(X_train.to(device))[:, -1, :].detach().cpu().numpy()
            # shift test predictions for plotting
            test_plot = np.ones_like(tpd) * np.nan
            test_plot[train_size+lookback:len(tpd)] = model(X_test.to(device))[:, -1, :].detach().cpu().numpy()

    if PLOT:
        
        plt.plot(tpd,c='b',label="real data")
        plt.plot(train_plot, c='r',label="traning")
        plt.plot(test_plot, c='g',label="testing")
        plt.xlabel("Sample (#)")
        plt.ylabel("Throughput (Mb/s)")
        plt.legend()
        plt.show()
    
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(),f"models/model_10_speed_{speed}_range_{base_station_range}.pt")

if __name__ == "__main__":
   
    cfg = load_config('config/config.yaml') 

    speeds = cfg["simulation"]["speed"]["index"] 
    base_station_range = cfg["simulation"]["base_station"]["range"]

    for speed in speeds:

        tpu, tpd = load_tp(speed=speed,
                           data_path=f"data/processed/speed{speed}/{base_station_range}")
        train(speed=speed,
              base_station_range=base_station_range)
