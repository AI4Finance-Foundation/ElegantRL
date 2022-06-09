## ElegantRL HelloWorld tutorial

The following three algorithms are 
- Deep Q-Network (DQN): the first DRL algorithm for discrete action space.
- Deep Deterministic Policy Gradient (DDPG): the first DRL algorithm for continuous action space.
- Proximal Policy Gradient (PPO): a popular DRL algorithm for continuous action space.


`ElegantRL Helloworld` are made simple.
- **Less lines of code**. (code lines <1000)
- **Less packages requirements**. (only `torch` and `gym` )
- **keep a consistent style with the full version of ElegantRL**.


## Download and run it.

1. Run the Jupyter Notebook. Open in Colab. [tutorial_helloworld_DQN_DDPG_PPO.ipynb](https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/tutorial_helloworld_DQN_DDPG_PPO.ipynb)

2. Run the Python files [tutorial_helloworld_DQN.py](https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl_helloworld/tutorial_helloworld_DQN.py), [ tutorial_helloworld_DDPG.py](https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl_helloworld/tutorial_helloworld_DDPG.py), [tutorial_helloworld_PPO.py](https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl_helloworld/tutorial_helloworld_PPO.py).
```
train_dqn_in_cartpole()
train_dqn_in_lunar_lander()

train_ddpg_in_pendulum()
train_ddpg_in_lunar_lander()
train_ddpg_in_bipedal_walker()

train_ppo_in_pendulum()
train_ppo_in_lunar_lander
train_ppo_in_bipedal_walker()
```

---


The files in `elegantrl_helloworld` are:
`config.py`, `agent.py`, `net.py`, `env.py`, `run.py`

![File_structure of ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL/raw/master/figs/File_structure.png)

One sentence summary: an agent `agent.py` with Actor-Critic networks `net.py` is trained `run.py` by interacting with an environment `env.py`.


In this tutorial, we only need to download the directory from [elegantrl_helloworld](https://github.com/AI4Finance-Foundation/ElegantRL/tree/master/elegantrl_helloworld) using the following code.


If you have some suggestions to ElegantRL Helloworld, we can discuss them in [ElegantRL issues/135: Suggestions for elegant_helloworld](https://github.com/AI4Finance-Foundation/ElegantRL/issues/135), and we will keep an eye on this issue.
ElegantRL's code, especially the Helloworld, really needs a lot of feedback to be better.
