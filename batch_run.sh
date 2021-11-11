#!/bin/bash
for idx in {1..3}
do
	for deploy in dist hydrodist hds
	do
		python train.py --wds anytown --obsrat 0.1 --adj binary --tag placement --runid $idx --deploy $deploy
		python train.py --wds ctown --obsrat 0.05 --adj binary --tag placement --runid $idx --deploy $deploy
		python train.py --wds richmond --obsrat 0.05 --adj binary --tag placement --runid $idx --deploy $deploy
	done
done
for idx in {1..3}
do
	python train.py --wds anytown --obsrat 0.1 --adj binary --tag placement --runid $idx --deploy hds --aversion 1
	python train.py --wds ctown --obsrat 0.05 --adj binary --tag placement --runid $idx --deploy hds --aversion 5
	python train.py --wds richmond --obsrat 0.05 --adj binary --tag placement --runid $idx --deploy hds --aversion 7
done
for idx in {1..15}
do
	python train.py --wds anytown --obsrat 0.1 --adj binary --tag placement --runid $idx --deploy random --deterministic
	python train.py --wds ctown --obsrat 0.05 --adj binary --tag placement --runid $idx --deploy random --deterministic
	python train.py --wds richmond --obsrat 0.05 --adj binary --tag placement --runid $idx --deploy random --deterministic
done
