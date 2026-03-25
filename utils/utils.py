import yaml 

import pandas as pd
import argparse

from os import mkdir

def load_base_stations_positions(positions_file="communication/base_stations.csv"):

    base_stations_positions = pd.read_csv(positions_file)

    return [ item for item in base_stations_positions.to_numpy() ], base_stations_positions.shape[0]

    
def load_config(file_name):

    with open(file_name,"r") as reader:
    
        return yaml.safe_load(reader)
