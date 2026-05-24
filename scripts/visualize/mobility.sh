execution=0
cars=$(yq '.simulation.cars' "config/config.yaml")
speeds=$(yq '.simulation.speed.index[]' "config/config.yaml") 

speed=${speeds[0]}

python -m utils.process.poi

sumo-gui -n mobility/raw/scenarios/$speed/manhattan_net_2.xml -r mobility/raw/scenarios/$speed/Krauss/$cars/flows_file_Krauss_"$cars"_"$execution".xml -a mobility/map/poi.xml,mobility/raw/scenarios/$speed/Krauss/$cars/manhatan_rerouter_Krauss_100_0.add.xml
