#!/bin/bash
listOfCameras="configs/cameras/americold/modesto/e_dock_north/ch12.yaml
configs/cameras/americold/modesto/e_dock_north/ch22.yaml"

for cameraName in $listOfCameras; do
	cameraFriendly=$(echo "$cameraName" | cut -d '/' -f5)
	cameraChannel=$(echo "$cameraName" | cut -d '/' -f6 | cut -d '.' -f1)
	sessionName="$cameraFriendly""_""$cameraChannel"
	echo "$sessionName"
	tmux new-session -d -s "$sessionName"
	tmux send-keys -t "$sessionName" "./experimental/tooling/run-prod.sh $cameraName" C-m
done
