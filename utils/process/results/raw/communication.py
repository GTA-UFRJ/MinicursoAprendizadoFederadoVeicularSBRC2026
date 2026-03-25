import csv

import numpy as np
import matplotlib.pyplot as plt
from os import listdir

import threading
import sys 

from utils.utils import (
        load_base_stations_positions,
        load_config)

# load configs
cfg = load_config("configs/config.yaml")

# Base station position
base_stations_file = cfg["simulation"]["base_station"]["positions"]
base_stations, n_base_stations = load_base_stations_positions(base_stations_file)

# Constants
c = 3e8  # Speed of light (m/s)
frequency = 5.9e9  # Carrier frequency for V2X (Hz)
lambda_ = c / frequency  # Wavelength (m)
shadowing_std = 3  # Shadowing standard deviation (dB), typical for V2X
shadowing_std_urban = 3  # Shadowing standard deviation (dB), typical for V2X => For the moment it is the same as highway. Perhaps to be adapted
shadowing_std_highway = 3  # Shadowing standard deviation (dB), typical for V2X => For the moment it is the same as Urban. Perhaps to be adapted
tx_power_bs = 46  # Base station transmit power in dBm. To be checked
tx_power_ue = 23  # UE transmit power in dBm. To be checked
total_bandwidth_dl = 20e6  # Total bandwidth for downlink (Hz)
total_bandwidth_ul = 10e6  # Total bandwidth for uplink (Hz)

# Path loss constants
V2X_LOS_pathloss_exp = 2.0  # Path loss exponent for LOS
V2X_NLOS_penalty = 20  # Additional path loss penalty for NLOS (dB). The NLOS is only a penalty. I think it is ok for this work since it is not the main focus.


# Bandwidth allocation methods. I made this to test different strategies at the base station level
bandwidth_allocation_method_dl = "proportional"  # Downlink allocation method
bandwidth_allocation_method_ul = "equal"  # Uplink allocation method

def read_input_file(filepath):
    """
    Read mobility data from an input file.
    Each line: {timestamp} {node_id} {x} {y} {speed}

    Parameters:
        filepath (str): Path to the input file.

    Returns:
        list: Parsed input data.
    """
    data = []
    with open(filepath, "r") as file:
        for line in file:
            parts = line.strip().split()
            timestamp, node_id, x, y, speed = (
                float(parts[0]),
                int(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
            data.append((timestamp, node_id, x, y, speed))
    return data

def path_loss_v2x(distance, los=True, urban=True):
    """
    Calculate path loss based on V2X models for LOS and NLOS.

    Parameters:
        distance (float): Distance between transmitter and receiver (meters).
        los (bool): Line-of-sight condition (True for LOS, False for NLOS).

    Returns:
        float: Path loss in dB.
    """
    if urban:
        if los:
            path_loss = 38.77 + 16.7 * np.log10(distance) + 18.2 * np.log10(frequency / 1e9)
        else:
            path_loss = 38.77 + 16.7 * np.log10(distance) + 18.2 * np.log10 (frequency / 1e9) + V2X_NLOS_penalty
        # Add shadowing
        shadowing = np.random.normal(0, shadowing_std)
    else:
        if los:
            path_loss = 32.4 + 20 * np.log10(distance) + 20 * np.log10(frequency / 1e9)
        else:
            path_loss = 32.4 + 20 * np.log10(distance) + 20 * np.log10(frequency / 1e9) + V2X_NLOS_penalty
        # Add shadowing
        shadowing = np.random.normal(0, shadowing_std)
    
    return path_loss + shadowing



def rician_fading(k_factor=8):
    """
    Generate Rician fading samples.

    Parameters:
        k_factor (float): Rician K-factor.

    Returns:
        float: Fading gain.
    """
    s = np.sqrt(k_factor / (k_factor + 1))  # LOS component
    sigma = np.sqrt(1 / (2 * (k_factor + 1)))  # NLOS component
    return np.abs(np.random.normal(s, sigma) + 1j * np.random.normal(0, sigma))

def equal_bandwidth_allocation(vehicles, total_bandwidth, link_type):
    """
    Allocate equal bandwidth to all vehicles for a specific link type.

    Parameters:
        vehicles (list): List of Vehicle objects.
        total_bandwidth (float): Total bandwidth to allocate.
        link_type (str): "uplink" or "downlink".
    """
    num_vehicles = len(vehicles)
    for vehicle in vehicles:
        if link_type == "downlink":
            vehicle.allocated_bandwidth_dl = total_bandwidth / num_vehicles
        elif link_type == "uplink":
            vehicle.allocated_bandwidth_ul = total_bandwidth / num_vehicles

def proportional_bandwidth_allocation(vehicles, total_bandwidth, link_type):
    """
    Allocate bandwidth proportionally to spectral efficiency for a specific link type.

    Parameters:
        vehicles (list): List of Vehicle objects.
        total_bandwidth (float): Total bandwidth to allocate.
        link_type (str): "uplink" or "downlink".
    """
    total_spectral_efficiency = sum(
        vehicle.spectral_efficiency_ul if link_type == "uplink" else vehicle.spectral_efficiency_dl
        for vehicle in vehicles
    )
    for vehicle in vehicles:
        if total_spectral_efficiency > 0:
            if link_type == "uplink":
                vehicle.allocated_bandwidth_ul = (
                    vehicle.spectral_efficiency_ul / total_spectral_efficiency
                ) * total_bandwidth
            else:
                vehicle.allocated_bandwidth_dl = (
                    vehicle.spectral_efficiency_dl / total_spectral_efficiency
                ) * total_bandwidth


def calculate_useful_throughput(raw_throughput):
    """
    Calculate useful throughput considering 5G protocol overheads.

    Equation:
    $$T_{useful} = T_{raw} \times (1 - O_{total})$$

    Parameters:
        raw_throughput (float): Raw throughput (Mbps).

    Returns:
        float: Useful throughput (Mbps).
    """
    # Overhead percentages for each layer
    overhead_phy = 0.10  # 10% PHY overhead
    overhead_mac = 0.08  # 8% MAC overhead
    overhead_pdcp = 0.05  # 5% PDCP overhead
    overhead_sdap = 0.03  # 3% SDAP overhead
    overhead_ip = 0.05  # 5% IP overhead

    # Total overhead fraction
    total_overhead = overhead_phy + overhead_mac + overhead_pdcp + overhead_sdap + overhead_ip

    # Useful throughput calculation
    useful_throughput = raw_throughput * (1 - total_overhead)
    return useful_throughput


class Vehicle:
    """
    Represents a vehicle in the V2X simulation.
    """

    def __init__(self, node_id, x, y, speed, los_prob=0.8):
        
        self.node_id = node_id
        self.position = np.array([x, y])
        self.speed = speed  # Speed in m/s
        self.los = np.random.rand() < los_prob
        self.path_loss_dl = None
        self.path_loss_ul = None
        
        self.rx_power_dl = None
        self.rx_power_ul = None
        
        self.spectral_efficiency_dl = None
        self.spectral_efficiency_ul = None
        
        self.throughput_dl = 0.0
        self.throughput_ul = 0.0
        
        self.useful_throughput_dl = 0.0
        self.useful_throughput_ul = 0.0
        
        self.allocated_bandwidth_dl = 0.0
        self.allocated_bandwidth_ul = 0.0
        
        self.distance_dl = 0.0
        self.distance_ul = 0.0 # The same as dl
        
        self.sinr_dl = 0.0
        self.sinr_ul = 0.0
        
        self.fading_dl = None
        self.fading_ul = None

        self.base_station_range = cfg["simulation"]["base_station"]["range"]

        self.base_station_position = self.attach_to_base_station()
    
    # attach a client to the closest base station
    def attach_to_base_station(self):

        closest_base_station = base_stations[0]
        shortest_distance = np.linalg.norm(self.position - closest_base_station)

        for index in range(1, len(base_stations)):

            # compute the distance from base station to the vehicle
            new_base_station_distance = np.linalg.norm(self.position - base_stations[index])

            # verify if is closer than the first one
            if new_base_station_distance < shortest_distance:
                
                closest_base_station = base_stations[index]
                shortest_distance = new_base_station_distance

        return closest_base_station

    def update_position(self, new_x, new_y):
        
        # disconnects the client from a previous base station
        if (not self.throughput_dl) or (not self.throughput_ul):
           
            self.attach_to_base_station()

        self.position = np.array([new_x, new_y])

    def calculate_downlink_metrics(self):
        """
        Calculate downlink metrics for the vehicle.

        Steps:
        1. Compute the distance to the base station.
        2. Calculate path loss using LOS or NLOS model.
        3. Model fading using Rician or Rayleigh distribution.
        4. Calculate received power using the transmit power and path loss.
        5. Compute SINR based on received power and noise.
        6. Calculate spectral efficiency using Shannon's formula.

        Updates:
            self.distance_dl (float): Distance to the base station.
            self.path_loss_dl (float): Path loss in dB.
            self.rx_power_dl (float): Received power in dBm.
            self.spectral_efficiency_dl (float): Spectral efficiency in bps/Hz.
        """
        # Step 1: Compute distance
        distance = np.linalg.norm(self.position - self.base_station_position)

        # Step 2: Calculate path loss
        self.path_loss_dl = path_loss_v2x(distance, self.los)

        # Step 3: Model fading
        self.fading_dl = rician_fading() if self.los else np.random.rayleigh()

        # Step 4: Calculate received power
        self.rx_power_dl = tx_power_bs - self.path_loss_dl + 10 * np.log10(self.fading_dl**2)

        # Step 5: Compute SINR
        noise_power = -174 + 10 * np.log10(total_bandwidth_dl)
        self.sinr_dl = 10 ** ((self.rx_power_dl - noise_power) / 10)

        # Step 6: Calculate spectral efficiency
        self.spectral_efficiency_dl = np.log2(1 + self.sinr_dl)

        # Update distance for reporting
        self.distance_dl = distance

    def calculate_uplink_metrics(self):
        """
        Calculate uplink metrics for the vehicle.
        IMPORTANT: NO INTEREFERENCES BETWEEN UE IS CONSIDERED.

        Steps:
        1. Compute the distance to the base station.
        2. Calculate path loss using LOS or NLOS model.
        3. Model fading using Rician or Rayleigh distribution.
        4. Calculate received power using the transmit power and path loss.
        5. Compute SINR based on received power and noise.
        6. Calculate spectral efficiency using Shannon's formula.

        Updates:
            self.distance_ul (float): Distance to the base station (meters).
            self.path_loss_ul (float): Path loss (dB).
            self.rx_power_ul (float): Received power at the base station (dBm).
            self.spectral_efficiency_ul (float): Spectral efficiency (bps/Hz).
        """
        # Step 1: Compute distance
        distance = np.linalg.norm(self.position - self.base_station_position)

        # Step 2: Calculate path loss
        self.path_loss_ul = path_loss_v2x(distance, self.los)

        # Step 3: Model fading
        self.fading_ul = rician_fading() if self.los else np.random.rayleigh()

        # Step 4: Calculate received power
        self.rx_power_ul = tx_power_ue - self.path_loss_ul + 10 * np.log10(self.fading_ul**2)

        # Step 5: Compute SINR
        noise_power = -174 + 10 * np.log10(total_bandwidth_ul)  # Noise in dBm
        self.sinr_ul = 10 ** ((self.rx_power_ul - noise_power) / 10)  # SINR in linear scale

        # Step 6: Calculate spectral efficiency
        self.spectral_efficiency_ul = np.log2(1 + self.sinr_ul)
        
        # Update distance
        self.distance_ul = distance

    # adding the disconnection condition
    def calculate_throughput(self):

        if self.distance_ul < self.base_station_range:
        
            self.throughput_dl = (self.spectral_efficiency_dl * self.allocated_bandwidth_dl) * 1e-6
            self.throughput_ul = (self.spectral_efficiency_ul * self.allocated_bandwidth_ul) * 1e-6
        
        else:

            self.throughput_dl = (self.spectral_efficiency_dl * self.allocated_bandwidth_dl) * 1e-6 * (self.base_station_range / 2*self.distance_ul)
            self.throughput_ul = (self.spectral_efficiency_ul * self.allocated_bandwidth_ul) * 1e-6 * (self.base_station_range / 2*self.distance_ul)

# save to an output file
def save_simulation_results_to_file(filename, vehicles, timestamp):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for vehicle in vehicles.values():
            writer.writerow([
                timestamp, vehicle.node_id, vehicle.position[0], vehicle.position[1],
                vehicle.speed, vehicle.distance_dl, vehicle.distance_ul, vehicle.throughput_dl,
                vehicle.throughput_ul, vehicle.spectral_efficiency_dl, vehicle.spectral_efficiency_ul,
                vehicle.fading_dl, vehicle.fading_ul, vehicle.useful_throughput_dl, vehicle.useful_throughput_ul
            ])
            
def simulate_v2x(input_data, filename="data/v2x_simulation.csv"):
    vehicles = {}

    # Create and initialize CSV file with headers
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Timestamp", "Node ID", "X Position", "Y Position", "Speed", "Distance DL",
            "Distance UL", "Throughput DL", "Throughput UL", "Spectral Efficiency DL",
            "Spectral Efficiency UL", "Fading DL", "Fading UL", "Useful Throughput DL", "Throughput UL" 
        ])
    # read the input data
    for _, node_id, x, y, speed in input_data:
        if node_id not in vehicles:
            vehicles[node_id] = Vehicle(node_id, x, y, speed)

    # start the simulation
    current_time = 0.0
    for timestamp, node_id, x, y, speed in input_data:
        if timestamp > current_time:
            # TODO: This part is not optimized since I am looping several times on vehicules!!!
            for vehicle in vehicles.values():
                vehicle.calculate_downlink_metrics()
                vehicle.calculate_uplink_metrics()
            equal_bandwidth_allocation(vehicles.values(), total_bandwidth_dl, "downlink")
            proportional_bandwidth_allocation(vehicles.values(), total_bandwidth_ul, "uplink")
            for vehicle in vehicles.values():
                vehicle.calculate_throughput()                
                vehicle.useful_throughput_dl = calculate_useful_throughput(vehicle.throughput_dl)
                vehicle.useful_throughput_ul = calculate_useful_throughput(vehicle.throughput_ul)
            # save the results for all vehicules.
            save_simulation_results_to_file(filename, vehicles, timestamp)
            current_time = timestamp
        vehicles[node_id].update_position(x, y)
        vehicles[node_id].speed = speed


if __name__ == "__main__":
    # Run the simulation
    # for file_name in listdir("mobility/raw"):
    
    #     input_data = read_input_file("mobility/raw/"+file_name)
    
    #     for index in range(30):
    #         simulate_v2x(input_data,"data/raw/"+file_name[:-4]+"_v2x_simulation_"+str(index)+".csv")
    
    THREAD = False
    SINGLE = True
    speed = 2 
    

    if THREAD:

        threads = {}

        for mobility in range(10):
            print("processing mobility file ",mobility)
            file_name = "mobility_"+str(mobility)+"_speed_"+str(speed)+".txt"
            input_data = read_input_file("mobility/processed/"+file_name)

            for index in range(30):
                print("index ", index)
                threads[str(index)+str(mobility)] = threading.Thread(target=simulate_v2x,
                                                                 args=(input_data,
                                                                       "data/raw/"+file_name[:-4]+"_simulation_"+str(index)+".csv"))
                threads[str(index)+str(mobility)].start()
            
        for mobility in range(10):
            for index in range(30): 
                threads[str(index)+str(mobility)].join()
    
        print("processed finished")

    elif SINGLE:
        mobility = sys.argv[1]
        speed = sys.argv[2]
        index = sys.argv[3]
        print("processing mobility file ",mobility)
        file_name = "mobility_"+str(mobility)+"_speed_"+str(speed)+".txt"
        input_data = read_input_file("mobility/processed/"+file_name)
        print("index ", index)
        simulate_v2x(input_data,
                     "data/raw/"+file_name[:-4]+"_simulation_"+str(index)+".csv")

    else:
        for mobility in range(10):
            print("processing mobility file ",mobility)
            file_name = "mobility_"+str(mobility)+"_speed_"+str(speed)+".txt"
            input_data = read_input_file("mobility/processed/"+file_name)

            for index in range(30):
                print("index ", index)
                simulate_v2x(input_data,
                             "data/raw/"+file_name[:-4]+"_simulation_"+str(index)+".csv")
            
        print("processed finished")



