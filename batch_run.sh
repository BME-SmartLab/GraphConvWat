#!/bin/bash
for idx in {1..3}
do
	for deploy in dist hydrodist hds
	do
		python train.py --epoch 1000 --adj binary --tag placement --deploy $deploy --wds anytown --obsrat 0.1 --batch 200
		python train.py --epoch 1000 --adj binary --tag placement --deploy $deploy --wds ctown --obsrat 0.05 --batch 120
		python train.py --epoch 1000 --adj binary --tag placement --deploy $deploy --wds richmond --obsrat 0.05 --batch 50
	done
done
for idx in {1..3}
do
	python train.py --epoch 1000 --adj binary --tag placement --deploy hds --wds anytown --obsrat 0.1 --aversion 1 --batch 200
	python train.py --epoch 1000 --adj binary --tag placement --deploy hds --wds ctown --obsrat 0.05 --aversion 5 --batch 120
	python train.py --epoch 1000 --adj binary --tag placement --deploy hds --wds richmond --obsrat 0.05 --aversion 7 --batch 50
done
for idx in {1..15}
do
	python train.py --epoch 1000 --adj binary --tag placement --deploy random --deterministic --wds anytown --obsrat 0.1 --batch 200
	python train.py --epoch 1000 --adj binary --tag placement --deploy random --deterministic --wds ctown --obsrat 0.05 --batch 120
	python train.py --epoch 1000 --adj binary --tag placement --deploy random --deterministic --wds richmond --obsrat 0.05 --batch 50
done
