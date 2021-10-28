#!/bin/bash
#for outer_idx in {1..3}
#do
#	for deploy in dist hydrodist hds
#	do
#		python train.py --obsrat 0.1 --adj binary --epoch 1000 --deploy $deploy --tag placement --wds anytown --batch 200
#		python train.py --obsrat 0.05 --adj binary --epoch 1000 --deploy $deploy --tag placement --wds ctown --batch 120
#		python train.py --obsrat 0.05 --adj binary --epoch 1000 --deploy $deploy --tag placement --wds richmond --batch 50
#	done
#	for idx in {1..5}
#	do
#		python train.py --deterministic --obsrat 0.1 --adj binary --epoch 1000 --deploy random --tag placement --wds anytown --batch 200
#		python train.py --deterministic --obsrat 0.05 --adj binary --epoch 1000 --deploy random --tag placement --wds ctown --batch 120
#		python train.py --deterministic --obsrat 0.05 --adj binary --epoch 1000 --deploy random --tag placement --wds richmond --batch 50
#	done
#done

for idx in {1..3}
do
	for deploy in dist hydrodist hds
	do
		python test_Taylor_metrics_for_sensor_placement.py --wds anytown --obsrat 0.1 --adj binary --tag placement --runid $idx --deploy $deploy
		python test_Taylor_metrics_for_sensor_placement.py --wds ctown --obsrat 0.05 --adj binary --tag placement --runid $idx --deploy $deploy
		python test_Taylor_metrics_for_sensor_placement.py --wds richmond --obsrat 0.05 --adj binary --tag placement --runid $idx --deploy $deploy
	done
done
for idx in {1..15}
do
	python test_Taylor_metrics_for_sensor_placement.py --wds anytown --obsrat 0.1 --adj binary --tag placement --runid $idx --deploy random
	python test_Taylor_metrics_for_sensor_placement.py --wds ctown --obsrat 0.05 --adj binary --tag placement --runid $idx --deploy random
	python test_Taylor_metrics_for_sensor_placement.py --wds richmond --obsrat 0.05 --adj binary --tag placement --runid $idx --deploy random
done
