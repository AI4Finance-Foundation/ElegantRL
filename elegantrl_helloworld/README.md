## ElegantRL HelloWorld tutorial: DQN --> DDPG --> PPO

We suggest to following this order to quickly learn about RL:
- DQN (Deep Q Network), a basic RL algorithms in discrete action space.
- DDPG (Deep Deterministic Policy Gradient), a basic RL algorithm in continuous action space.
- PPO (Proximal Policy Gradient), a widely used RL algorithms in continuous action space.


We hope that the `ElegantRL Helloworld` would help people who want to learn about reinforcement learning to quickly run a few introductory examples.
- **Less lines of code**. (code lines <1000)
- **Less packages requirements**. (only `torch` and `gym` )
- **keep a consistent style with the full version of ElegantRL**.


## ElegantRL HelloWorld tutorial: Download and run it.

# todo 2022-04-20 17:54:34




The files in `elegantrl_helloworld` including:
`config.py`, `agent.py`, `net.py`, `env.py`, `run.py`

![File_structure of ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL/raw/master/figs/File_structure.png)

One sentence summary: an agent `agent.py` with Actor-Critic networks `net.py` is trained `run.py` by interacting with an environment `env.py`.


In this tutorial, we only need to download the directory from [elegantrl_helloworld](https://github.com/AI4Finance-Foundation/ElegantRL/tree/master/elegantrl_helloworld) using the following code.


If you have any suggestion about ElegantRL Helloworld, you can discuss them in [ElegantRL issues/135: Suggestions for elegant_helloworld](https://github.com/AI4Finance-Foundation/ElegantRL/issues/135), and we will keep an eye on this issue.
ElegantRL's code, especially the Helloworld, really needs a lot of feedback to be better.
