#!/bin/bash
listOfCameras="configs/cameras/rite_hite/hansen/door_22/ee117015-fcfe-c17b-d775-d33a3ef682ca.yaml
configs/cameras/rite_hite/hansen/egg_aisle/13af56ad-17db-b29f-6195-ded9e034b41d.yaml
configs/cameras/rite_hite/hansen/unknown/6ef568df-f4fb-0199-d775-cab1d1d14a4b.yaml
configs/cameras/rite_hite/hansen/unknown/a5f2184d-620d-3f91-f590-949e0cddd63d.yaml
configs/cameras/rite_hite/hansen/unknown/aadb2c08-e26c-929c-522c-7a9aaef5645f.yaml
configs/cameras/rite_hite/hansen/unknown/d4978fd3-eec4-5970-a108-07f6740241c4.yaml"

for cameraName in $listOfCameras; do
	cameraFriendly=$(echo "$cameraName" | cut -d '/' -f5)
	cameraChannel=$(echo "$cameraName" | cut -d '/' -f6 | cut -d '.' -f1)
	sessionName="$cameraFriendly""_""$cameraChannel"
	echo "$sessionName"
	tmux new-session -d -s "$sessionName"
	tmux send-keys -t "$sessionName" "./experimental/tooling/run-prod.sh $cameraName" C-m
done
