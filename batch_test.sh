#!/bin/bash
for wds in anytown ctown richmond
do
	for deploy in random dist hydrodist hds
	do
		echo $wds - $deploy
		python train.py --wds $wds --deploy $deploy --tag alma
	done
done
