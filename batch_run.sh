#!/bin/bash
for idx in {1..5}
do
	python train.py --deterministic --obsrat 0.1 --adj binary --epoch 1000 --deploy random --tag placement --wds anytown --batch 200
	python train.py --deterministic --obsrat 0.05 --adj binary --epoch 1000 --deploy random --tag placement --wds ctown --batch 120
	python train.py --deterministic --obsrat 0.05 --adj binary --epoch 1000 --deploy random --tag placement --wds richmond --batch 50
done
for deploy in dist hydrodist hds
do
	python train.py --deterministic --obsrat 0.1 --adj binary --epoch 1000 --deploy $deploy --tag placement --wds anytown --batch 200
	python train.py --deterministic --obsrat 0.05 --adj binary --epoch 1000 --deploy $deploy --tag placement --wds ctown --batch 120
	python train.py --deterministic --obsrat 0.05 --adj binary --epoch 1000 --deploy $deploy --tag placement --wds richmond --batch 50
done
