#!/bin/bash

g_type=clustered

result_root=results/dqn-$g_type

test_min_n=15
test_max_n=20

# max belief propagation iteration
max_bp_iter=4

# embedding size
embed_dim=64

# gpu card id
dev_id=0

# max batch size for training/testing
batch_size=128

net_type=QNet
decay=0.1

# set reg_hidden=0 to make a linear regression
reg_hidden=32

# learning rate
learning_rate=0.0001

# init weights with rand normal(0, w_scale)
w_scale=0.01

# nstep
n_step=1

knn=10

min_n=15
max_n=20

num_env=1
mem_size=50000

max_iter=200000

# folder to save the trained model
save_dir=$result_root/ntype-$net_type-embed-$embed_dim-nbp-$max_bp_iter-rh-$reg_hidden

python evaluate.py \
    -net_type $net_type \
    -dev_id $dev_id \
    -n_step $n_step \
    -data_root ../../data/tsp2d \
    -decay $decay \
    -knn $knn \
    -test_min_n $test_min_n \
    -test_max_n $test_max_n \
    -min_n $min_n \
    -max_n $max_n \
    -num_env $num_env \
    -max_iter $max_iter \
    -mem_size $mem_size \
    -g_type $g_type \
    -learning_rate $learning_rate \
    -max_bp_iter $max_bp_iter \
    -net_type $net_type \
    -max_iter $max_iter \
    -save_dir $save_dir \
    -embed_dim $embed_dim \
    -batch_size $batch_size \
    -reg_hidden $reg_hidden \
    -momentum 0.9 \
    -l2 0.00 \
    -w_scale $w_scale
