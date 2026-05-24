import os

from pickle import load
import matplotlib.pyplot as plt
from numpy import mean, std

from utils.utils import load_config 
from .legends import (
    legends_dicts,
    colors,
    style,
    FONTSIZE,
    LEGEND_FONTSIZE,
    FIGURE_SIZE
)

def selection_error_plot(file_path:str="results/client_selection",
                         model_size:int=500, 
                         base_station_range:int=2000,
                         PLOT:bool=False, 
                         n_executions:int=10, 
                         n_selected:list=[1,100],
                         language:str="pt",
                         servers:list=["random",
                                       "m_fastest",
                                       "tofl_oracle",
                                       "tofl_estimator_dl",
                                       "tofl_estimator_m_fastest"]):
    
    plt.figure(figsize=FIGURE_SIZE)
    
    legends = legends_dicts[language]

    if language == "en":
        
        plt.xlabel("Selected Clients (#)", fontsize=FONTSIZE)
        plt.ylabel("Total Training Time (s)", fontsize=FONTSIZE)

    elif language == "pt":

        plt.xlabel("Quantidade de Clientes Selecionados (#)", fontsize=FONTSIZE)
        plt.ylabel("Tempo Total de Treinamento (s)", fontsize=FONTSIZE)


    results = { server : [ ] 
               for server in servers }
    
    
    for server in servers:

        for dataset in range(n_executions):
        
            with open(f"{file_path}/model_{server}_size_{model_size}_dataset_{dataset}","rb") as loader:
            
                result_list = load(loader)
                results[server].append(result_list)
    
    for server in servers:
        
        m = mean(results[server],axis=0)
        s = std(results[server],axis=0)
        x = n_selected

        plt.plot(x,
                 m,
                 linewidth=3,
                 label=legends[server],
                 color=colors[server],
                 linestyle=style[server])

        plt.fill_between(x,
                         m - s,
                         m + s,
                         color=colors[server],
                         alpha=0.2)

    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    
    if base_station_range >= 1400:

        plt.ylim(0,70)

    else:

        plt.ylim(0,1000)
    
    figure_path = "figures/communication"

    os.makedirs(figure_path, 
                exist_ok=True)

    plt.savefig(f"{figure_path}/model_size_{model_size}_{base_station_range}_{language}.pdf",
                dpi=300,
                bbox_inches='tight')


    if PLOT:
        
        plt.show()
    
    plt.close()

if __name__ == "__main__":

    cfg = load_config('configs/config.yaml')
    
    model_size = cfg["simulation"]["model"]["size"]
    speeds = cfg["simulation"]["speed"]["index"]
    servers = cfg["simulation"]['federated_learning']['server']["strategy"]
    repetitions = cfg["simulation"]["mobility"]["repetitions"]
    rounds = cfg["simulation"]["federated_learning"]["server"]["rounds"]    
    base_station_range = cfg['simulation']['base_station']['range']
    n_selected_clients_list = cfg["simulation"]["federated_learning"]["server"]["n_clients_selected"]

    for lg in ["pt", "en"]:

        for speed in speeds:

            selection_error_plot(f"results/client_selection/processed/speed{speed}/{base_station_range}", 
                                 model_size, 
                                 base_station_range=base_station_range,
                                 servers=servers,
                                 n_executions=repetitions,
                                 n_selected=n_selected_clients_list,
                                 language=lg)
        
