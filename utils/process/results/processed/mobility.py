import threading
from utils.utils import load_config

def process_scenario(n_nodes=20, 
                     file_name="mobility/raw/scenarios/0/Krauss/20/manhattan_Krauss_20_",
                     execution=0,
                     speed=0):

    file_name = file_name+str(execution)+".tcl"
    processed_file = "mobility/processed/mobility_"+str(execution)+"_speed_"+str(speed)+".txt"
    txt_writer = ""

    with open(file_name, "r") as reader:
        current_nodes = 0
        start_process = False
        
        for line in reader:
            if current_nodes < n_nodes:
            
                if ") set Z_" in line:
                    current_nodes += 1
            elif current_nodes == n_nodes and not start_process:
                time = float(line.split(' ')[2])
                start_process = True
            
            elif start_process:
                if float(line.split(' ')[2]) <= time:
                    pass

                else:
                    elements_list = line.split(' ')
                    txt_writer += elements_list[2] + ' ' + \
                                elements_list[3][elements_list[3].find('(')+1:elements_list[3].find(')')] + ' ' + \
                                elements_list[5] + ' ' + \
                                elements_list[6] + ' ' + \
                                elements_list[7][:-2] + "\n"
                    
    with open(processed_file, "w") as writer:
        writer.writelines(txt_writer)

if __name__ == "__main__":

    cfg = load_config("configs/config.yaml")

    n_cars = str(cfg["simulation"]["cars"])
    speeds = cfg["simulation"]["speed"]["index"]
    repetitions = cfg["simulation"]["mobility"]["repetitions"]
    threads = {}

    for speed in speeds:

        for execution in range(repetitions):
        
            threads["thread"+str(execution)+str(speed)] = threading.Thread(target=process_scenario,
                                                            args=(int(n_cars), 
                                                                  f"mobility/raw/scenarios/{speed}/Krauss/{n_cars}/manhattan_Krauss_{n_cars}_",
                                                                  execution,
                                                                  speed))

            threads["thread"+str(execution)+str(speed)].start()
        
            threads["thread"+str(execution)+str(speed)].join()


    print("process finished")

