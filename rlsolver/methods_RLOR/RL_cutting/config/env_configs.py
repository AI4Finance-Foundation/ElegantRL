starter_config = {
    "load_dir"        : 'instances/randomip_n10_m20',   # this is the location of the randomly generated instances (you may specify a different directory)
    "idx_list"        : list(range(50,60)),                # take the first 20 instances from the directory
    "timelimit"       : 10,                             # the maximum horizon length is 50
    "reward_type"     : 'obj'                           # DO NOT CHANGE reward_type
}

starter_config2 = {
"load_dir"        : 'instances/randomip_n15_m15',   # this is the location of the randomly generated instances (you may specify a different directory)
    "idx_list"        : list(range(5)),                # take the first 20 instances from the directory
    "timelimit"       : 20,                             # the maximum horizon length is 50
    "reward_type"     : 'obj'                           # DO NOT CHANGE reward_type
}


veryeasy_config = {
    "load_dir"        : 'instances/randomip_n25_m25',   # this is the location of the randomly generated instances (you may specify a different directory)
    "idx_list"        : list(range(1)),                # take the first 20 instances from the directory
    "timelimit"       : 20,                             # the maximum horizon length is 50
    "reward_type"     : 'obj'                           # DO NOT CHANGE reward_type
}

# Easy Setup: Use the following environment settings. We will evaluate your agent with the same easy config below:
easy_config = {
    "load_dir"        : 'instances/train_10_n60_m60',
    "idx_list"        : list(range(10)),
    "timelimit"       : 50,
    "reward_type"     : 'obj'
}

# Hard Setup: Use the following environment settings. We will evaluate your agent with the same hard config below:
hard_config = {
    "load_dir"        : 'instances/train_100_n60_m60',
    "idx_list"        : list(range(99)),
    "timelimit"       : 50,
    "reward_type"     : 'obj'
}

test_config = {
    "load_dir"        : 'instances/test_100_n60_m60',
    "idx_list"        : list(range(10)),
    "timelimit"       : 50,
    "reward_type"     : 'obj'

}
