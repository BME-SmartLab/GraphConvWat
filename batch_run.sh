#!/bin/bash
for idx in {1..22}
do
	python train.py --obsrat 0.1 --adj binary --epoch 1000 --deploy random --tag placement --wds anytown --batch 200 --idx $idx
done
#for deploy in random dist hydrodist hds
#do
#	python train.py --deterministic --obsrat 0.1 --adj binary --epoch 1000 --deploy $deploy --tag placement --wds anytown --batch 200
#done
#for deploy in random dist hydrodist hds
#do
#	python train.py --deterministic --obsrat 0.05 --adj binary --epoch 1000 --deploy $deploy --tag placement --wds ctown --batch 100
#done
#for deploy in random dist hydrodist hds
#do
#	python train.py --deterministic --obsrat 0.05 --adj binary --epoch 1000 --deploy $deploy --tag placement --wds richmond --batch 40
#done
