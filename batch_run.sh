#!/bin/bash
for idx in {1..3}
do
	for deploy in dist hydrodist hds
	do
		python train.py --wds anytown --obsrat 0.1 --adj binary --tag placement --deploy $deploy --batch 200
		python train.py --wds ctown --obsrat 0.05 --adj binary --tag placement --deploy $deploy --batch 120
		python train.py --wds richmond --obsrat 0.05 --adj binary --tag placement --deploy $deploy --batch 50
	done
done
for idx in {1..3}
do
	python train.py --wds anytown --obsrat 0.1 --adj binary --tag placement --deploy hds --aversion 1 --batch 200
	python train.py --wds ctown --obsrat 0.05 --adj binary --tag placement --deploy hds --aversion 5 --batch 120
	python train.py --wds richmond --obsrat 0.05 --adj binary --tag placement --deploy hds --aversion 7 --batch 50
done
for idx in {1..15}
do
	python train.py --wds anytown --obsrat 0.1 --adj binary --tag placement --deploy random --deterministic --batch 200
	python train.py --wds ctown --obsrat 0.05 --adj binary --tag placement --deploy random --deterministic --batch 120
	python train.py --wds richmond --obsrat 0.05 --adj binary --tag placement --deploy random --deterministic --batch 50
done
