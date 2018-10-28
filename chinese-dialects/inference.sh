#! /bin/sh

mkdir /result
cd /inference
bash run.sh inference.py ../dataset models.pkl ../result/result.txt
