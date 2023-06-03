#!/bin/bash
listOfCameras="configs/cameras/uscold/laredo/dock01/cha.yaml
configs/cameras/uscold/laredo/dock03/cha.yaml
configs/cameras/uscold/laredo/dock04/cha.yaml
configs/cameras/uscold/laredo/doors_14_20/cha.yaml
configs/cameras/uscold/laredo/room_f1/cha.yaml
configs/cameras/uscold/laredo/room_d1p/cha.yaml
configs/cameras/uscold/laredo/room_d2/cha.yaml
configs/cameras/uscold/laredo/room_d3/cha.yaml
configs/cameras/uscold/laredo/room_d4/cha.yaml
configs/cameras/uscold/laredo/room_d5p/cha.yaml"

for cameraName in $listOfCameras; do
	cameraFriendly=$(echo "$cameraName" | cut -d '/' -f5)
	cameraChannel=$(echo "$cameraName" | cut -d '/' -f6 | cut -d '.' -f1)
	sessionName="$cameraFriendly""_""$cameraChannel"
	echo "$sessionName"
	tmux new-session -d -s "$sessionName"
	tmux send-keys -t "$sessionName" "./experimental/tooling/run-prod.sh $cameraName" C-m
done
