#!/bin/bash

min_n=15
max_n=20
ep=0.05
f=0.2
output_root=../../../data/scp

if [ ! -e $output_root ];
then
    mkdir -p $output_root
fi

python gen_graph_only.py \
    -save_dir $output_root \
    -max_n $max_n \
    -min_n $min_n \
    -num_graph 1000 \
    -edge_prob 0.05 \
    -frac_primal 0.2
