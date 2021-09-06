#!/bin/bash
for wds in anytown ctown richmond
do
	for runid in {1..20}
	do
		python test_Taylor_metrics.py --wds $wds --adj binary --model interp --obsrat .8 --runid $runid
		python test_Taylor_metrics.py --wds $wds --adj binary --model interp --obsrat .4 --runid $runid
		python test_Taylor_metrics.py --wds $wds --adj binary --model interp --obsrat .2 --runid $runid
		python test_Taylor_metrics.py --wds $wds --adj binary --model interp --obsrat .1 --runid $runid
		python test_Taylor_metrics.py --wds $wds --adj binary --model interp --obsrat .05 --runid $runid
	done
done
