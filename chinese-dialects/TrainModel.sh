#! /bin/bash

if [[ $# -ne 2 ]]
then
	echo Usage: $0 data_dir model_path
	exit 1
fi

./run.sh python train.py $@
