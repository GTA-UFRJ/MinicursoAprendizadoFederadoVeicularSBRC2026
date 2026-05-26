execution=0
cars=$(yq '.simulation.cars' "configs/config.yaml")
speeds=$(yq '.simulation.speed.index[]' "configs/config.yaml") 

speed=${speeds[0]}

sumo-gui -n mobility/raw/scenarios/$speed/manhattan_net_2.xml -r mobility/raw/scenarios/$speed/Krauss/$cars/flows_file_Krauss_"$cars"_"$execution".xml -a mobility/raw/scenarios/$speed/Krauss/$cars/manhatan_rerouter_Krauss_"$cars"_0.add.xml
