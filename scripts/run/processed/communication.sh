#!/bin/bash

speeds=$(yq '.simulation.speed.index[]' "configs/config.yaml")
mobility=$(yq '.simulation.mobility.repetitions' "configs/config.yaml")

for speed in $speeds

do

	for index in $( seq 0 $(($mobility-1)))

	do

		python -m utils.process.results.processed.communication $speed $index &

	done

	wait
done

wait 

echo "process finished"

