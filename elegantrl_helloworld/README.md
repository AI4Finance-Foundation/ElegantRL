## ElegantRL HelloWorld tutorial: DQN --> DDPG --> PPO

We suggest to following this order to quickly learn about RL:
- DQN (Deep Q Network), a basic DRL algorithms in discrete action space.
- DDPG (Deep Deterministic Policy Gradient), a basic DRL algorithm in continuous action space.
- PPO (Proximal Policy Gradient), a widely used DRL algorithms in continuous action space.


We hope that the `ElegantRL Helloworld` would help people who to learn about reinforcement learning and quickly run a few introductory examples.
- **Less lines of code**. (code lines <1000)
- **Less packages requirements**. (only `torch` and `gym` )
- **keep a consistent style with the full version of ElegantRL**.


## Download and run it.

1. Run the Jupyter Notebook. Open in Colab. [tutorial_helloworld_DQN_DDPG_PPO.ipynb](https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/tutorial_helloworld_DQN_DDPG_PPO.ipynb)

2. Run the Python file [tutorial_helloworld_DQN_DDPG_PPO.py](https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl_helloworld/tutorial_helloworld_DQN_DDPG_PPO.py)
```
train_dqn_in_cartpole()
train_dqn_in_lunar_lander()

train_ddpg_in_pendulum()
train_ppo_in_pendulum()

train_ddpg_in_lunar_lander_or_bipedal_walker()
train_ppo_in_lunar_lander_or_bipedal_walker()
```

---


The files in `elegantrl_helloworld` including:
`config.py`, `agent.py`, `net.py`, `env.py`, `run.py`

![File_structure of ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL/raw/master/figs/File_structure.png)

One sentence summary: an agent `agent.py` with Actor-Critic networks `net.py` is trained `run.py` by interacting with an environment `env.py`.


In this tutorial, we only need to download the directory from [elegantrl_helloworld](https://github.com/AI4Finance-Foundation/ElegantRL/tree/master/elegantrl_helloworld) using the following code.


If you have any suggestion about ElegantRL Helloworld, you can discuss them in [ElegantRL issues/135: Suggestions for elegant_helloworld](https://github.com/AI4Finance-Foundation/ElegantRL/issues/135), and we will keep an eye on this issue.
ElegantRL's code, especially the Helloworld, really needs a lot of feedback to be better.
