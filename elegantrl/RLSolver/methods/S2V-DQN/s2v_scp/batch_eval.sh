#!/bin/bash

ep=0.05
min_n=1800
max_n=2000
dev_id=7
#for p in '200 300' '300 400' '400 500' '500 600' '1000 1200'; do
for p in '15 20' '40 50' '50 100' '100 200' '200 300' '400 500' '500 600' '1000 1200'; do
#	./template_run_eval.sh $p $ep $min_n $max_n $dev_id
	./template_run_eval.sh $min_n $max_n $ep $p $dev_id
done

