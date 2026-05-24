
import os

import matplotlib.pyplot as plt

from utils.utils import load_config

from ..estimator.data import (
    load_tp,
)


def plot_throughput(speed=0,
                    PLOT=True,
                    lang="pt"):

    if lang == "pt":

        plt.xlabel("Amostra (#)",fontsize=16)
        plt.ylabel("Vazão (Mb/s)",fontsize=16)
        

    elif lang == "en":
        
        plt.xlabel("Sample (#)",fontsize=16)
        plt.ylabel("Throughput (Mb/s)",fontsize=16)


    plt.plot(tpd,
             c='b')
        
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    figure_path = "figures/communication"

    os.makedirs(figure_path, 
                exist_ok=True)

    plt.savefig(f"{figure_path}/speed{speed}_{lang}.png",
                dpi=300,
                bbox_inches='tight')
    
    if PLOT:
        
        plt.show()
    
    
if __name__ == "__main__":
   
    cfg = load_config('configs/config.yaml') 

    speeds = cfg["simulation"]["speed"]["index"] 

    for speed in speeds:

        tpu, tpd = load_tp(speed=speed,
                           data_path=f"data/processed/speed{speed}/")
        
        plot_throughput(speed=speed)
