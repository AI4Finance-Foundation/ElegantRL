#!/bin/bash

result_root=results/dqn-meme

# max belief propagation iteration
max_bp_iter=4

# embedding size
embed_dim=64

# gpu card id
dev_id=0

# max batch size for training/testing
batch_size=64

net_type=QNet

# set reg_hidden=0 to make a linear regression
reg_hidden=64

# learning rate
learning_rate=0.0001

# init weights with rand normal(0, w_scale)
w_scale=0.01

# nstep
n_step=2

min_n=400
max_n=550
pq=10
num_env=10
mem_size=500000

max_iter=1000000
data_test=../../data/meme_setcover/meme-ntest-1000-pq-${pq}.pkl
# folder to save the trained model
save_dir=$result_root/embed-$embed_dim-nbp-$max_bp_iter-rh-$reg_hidden-pq-$pq

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python main.py \
    -data_test $data_test \
    -data_root ../../data/memetracker \
    -pq $pq \
    -dev_id $dev_id \
    -n_step $n_step \
    -min_n $min_n \
    -max_n $max_n \
    -num_env $num_env \
    -max_iter $max_iter \
    -mem_size $mem_size \
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
    -w_scale $w_scale \
    2>&1 | tee $save_dir/log-$min_n-${max_n}.txt
