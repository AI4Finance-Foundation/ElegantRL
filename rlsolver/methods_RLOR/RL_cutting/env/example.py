from env.gymenv_v2 import make_multiple_env
import numpy as np


import wandb
wandb.login()
run=wandb.init(project="finalproject", entity="ieor-4575", tags=["training-easy"])
#run=wandb.init(project="finalproject", entity="ieor-4575", tags=["training-hard"])
#run=wandb.init(project="finalproject", entity="ieor-4575", tags=["test"])

### TRAINING

# Setup: You may generate your own instances on which you train the cutting agent.
custom_config = {
    "load_dir"        : 'instances/randomip_n60_m60',   # this is the location of the randomly generated instances (you may specify a different directory)
    "idx_list"        : list(range(20)),                # take the first 20 instances from the directory
    "timelimit"       : 50,                             # the maximum horizon length is 50
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

if __name__ == "__main__":
    # create env
    env = make_multiple_env(**easy_config)

    for e in range(1):
        # gym loop
        s = env.reset()   # samples a random instance every time env.reset() is called
        d = False
        t = 0
        repisode = 0

        while not d:
            a = np.random.randint(0, s[-1].size, 1)            # s[-1].size shows the number of actions, i.e., cuts available at state s
            s, r, d, _ = env.step(list(a))
            print('episode', e, 'step', t, 'reward', r, 'action space size', s[-1].size, 'action', a[0])

            A, b, c0, cuts_a, cuts_b = s
            #print(A.shape, b.shape, c0.shape, cuts_a.shape, cuts_b.shape)

            t += 1
            repisode += r

	#wandb logging
        wandb.log({"Training reward" : repisode})
	#if using hard-config make sure to use "training-hard" tag in wandb.init in the initialization on top



