#!/bin/bash

data_root=../../data/meme_setcover
num=1000
pq=10

if [ ! -e $data_root ];
then
    mkdir -p $data_root
fi

python gen_data.py \
    -out_root $data_root \
    -data_root ../../data/memetracker \
    -num $num \
    -pq $pq
    

