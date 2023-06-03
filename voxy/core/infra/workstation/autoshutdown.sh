##
## Copyright 2020-2021 Voxel Labs, Inc.
## All rights reserved.
##
## This document may not be reproduced, republished, distributed, transmitted,
## displayed, broadcast or otherwise exploited in any manner without the express
## prior written permission of Voxel Labs, Inc. The receipt or possession of this
## document does not convey any rights to reproduce, disclose, or distribute its
## contents, or to manufacture, use, or sell anything that it may describe, in
## whole or in part.
##

#!/bin/bash
threshold=5
count=0
while true; do
	load=$(awk '{u=$2+$4; t=$2+$4+$5; if (NR==1){u1=u; t1=t;} else print ($2+$4-u1) * 10000 / (t-t1); }' <(grep 'cpu ' /proc/stat) <(
		sleep 60
		grep 'cpu ' /proc/stat
	))
	load2=$(printf "%.0f\n" "$load")
	echo "$load"
	echo "$load2"
	if [[ $load2 -lt $threshold ]]; then
		echo "Idling.."
		((count += 1))
	else
		((count = 0))
	fi
	echo "Idle minutes count = $count"
	if ((count > 120)); then
		echo Shutting down
		sleep 30
		history -a
		sudo poweroff
	fi
done
