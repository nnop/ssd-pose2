#!/bin/bash

while true
do
	echo "tyring"
	python examples/ssd/ssd_ObjNet.py --opt_dir=configs/sepfullpose.json
	sleep 1
done
