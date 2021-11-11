#!/bin/bash
for idx in {1..3}
do
	for deploy in dist hydrodist hds
	do
		python test_Taylor_metrics_for_sensor_placement.py --wds anytown --obsrat 0.1 --adj binary --tag placement --runid $idx --deploy $deploy
		python test_Taylor_metrics_for_sensor_placement.py --wds ctown --obsrat 0.05 --adj binary --tag placement --runid $idx --deploy $deploy
		python test_Taylor_metrics_for_sensor_placement.py --wds richmond --obsrat 0.05 --adj binary --tag placement --runid $idx --deploy $deploy
	done
done
for idx in {1..3}
do
	python test_Taylor_metrics_for_sensor_placement.py --wds anytown --obsrat 0.1 --adj binary --tag placement --runid $idx --deploy hds --aversion 1
	python test_Taylor_metrics_for_sensor_placement.py --wds ctown --obsrat 0.05 --adj binary --tag placement --runid $idx --deploy hds --aversion 5
	python test_Taylor_metrics_for_sensor_placement.py --wds richmond --obsrat 0.05 --adj binary --tag placement --runid $idx --deploy hds --aversion 7
done
for idx in {1..15}
do
	python test_Taylor_metrics_for_sensor_placement.py --wds anytown --obsrat 0.1 --adj binary --tag placement --runid $idx --deploy random --deterministic
	python test_Taylor_metrics_for_sensor_placement.py --wds ctown --obsrat 0.05 --adj binary --tag placement --runid $idx --deploy random --deterministic
	python test_Taylor_metrics_for_sensor_placement.py --wds richmond --obsrat 0.05 --adj binary --tag placement --runid $idx --deploy random --deterministic
done
