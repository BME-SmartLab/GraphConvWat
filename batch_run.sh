#!/bin/bash
#for idx in {1..3}
#do
#	for deploy in master dist hydrodist hds hdvar
#	do
#		python train.py --epoch 1000 --adj binary --tag ms --deploy $deploy --wds anytown --budget 1 --batch 200
#		python train.py --epoch 1000 --adj binary --tag ms --deploy $deploy --wds ctown --budget 5 --batch 120
#		python train.py --epoch 1000 --adj binary --tag ms --deploy $deploy --wds richmond --budget 10 --batch 50
#	done
#done
for idx in {1..15}
do
	python train.py --epoch 1000 --adj binary --tag ms --deploy xrandom --deterministic --wds anytown --budget 1 --batch 200
done
for idx in {1..15}
do
	python train.py --epoch 1000 --adj binary --tag ms --deploy xrandom --deterministic --wds ctown --budget 5 --batch 120
done
for idx in {1..15}
do
	python train.py --epoch 1000 --adj binary --tag ms --deploy xrandom --deterministic --wds richmond --budget 10 --batch 50
done
