## ElegantRL “小雅”: Massively Parallel Library for Cloud-native Deep Reinforcement Learning

[![Downloads](https://pepy.tech/badge/elegantrl)](https://pepy.tech/project/elegantrl)
[![Downloads](https://pepy.tech/badge/elegantrl/week)](https://pepy.tech/project/elegantrl)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/elegantrl.svg)](https://pypi.org/project/elegantrl/)

<br/>
<a href="https://github.com/AI4Finance-LLC/ElegantRL" target="\_blank">
	<div align="center">
		<img src="figs/icon.jpg" width="40%"/> 
	</div>
<!-- 	<div align="center"><caption>Slack Invitation Link</caption></div> -->
</a>
<br/>

ElegantRL ([website](https://elegantrl.readthedocs.io/en/latest/index.html)) is developed for practitioners with
the following advantages:

- **Cloud-native**: follows a cloud-native paradigm through microservice architecture and containerization, supporting [ElegantRL-Podracer](https://elegantrl.readthedocs.io/en/latest/tutorial/elegantrl-podracer.html) and [FinRL-Podracer](https://elegantrl.readthedocs.io/en/latest/tutorial/finrl-podracer.html).

- **Scalable**: fully exploits the parallelism of DRL algorithms at multiple levels, making it easily scale out to hundreds or thousands of computing nodes on a cloud platform, say, a [DGX SuperPOD platform](https://www.nvidia.com/en-us/data-center/dgx-superpod/) with thousands of GPUs.

- **Elastic**: allows to elastically and automatically allocate computing resources on the cloud.

- **Lightweight**: the core codes <1,000 lines (check [Elegantrl_Helloworld](https://github.com/AI4Finance-Foundation/ElegantRL/tree/master/elegantrl_helloworld)).

- **Efficient**: in many testing cases (single GPU/multi-GPU/GPU cloud), we find it more efficient than [Ray RLlib](https://github.com/ray-project/ray).

- **Stable**: much much much more stable than [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) by utilizing various ensemble methods.

ElegantRL implements the following model-free deep reinforcement learning (DRL) algorithms:

- **DDPG, TD3, SAC, PPO, REDQ** for continuous actions,
- **DQN, Double DQN, D3QN, SAC** for discrete actions,
- **QMIX, VDN, MADDPG, MAPPO, MATD3** for multi-agent environment.

For the details of DRL algorithms, please check out the educational
webpage [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/).

ElegantRL supports the following simulators:

- **Isaac Gym** for massively parallel simulation,
- **OpenAI Gym, MuJoCo, PyBullet, FinRL** for benchmarking.

“小雅”源于《诗经·小雅·鹤鸣》，旨在「他山之石，可以攻玉」。

## Contents

- [News](#News)
- [ElegantRL-Helloworld](#ElegantRL-Helloworld)
- [File Structure](#File-Structure)
- [Experimental Demos](#Experimental-Demos)
- [Requirements](#Requirements)
- [Citation](#Citation)


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AI4Finance-Foundation/ElegantRL&type=Date)](https://star-history.com/#AI4Finance-Foundation/ElegantRL&Date)


## News

- [MLearning.ai] [ElegantRL: Much More Stable Deep Reinforcement Learning Algorithms than Stable-Baseline3](https://medium.com/mlearning-ai/elegantrl-much-much-more-stable-than-stable-baseline3-f096533c26db), Mar 3, 2022.
- [Towardsdatascience] [ElegantRL-Podracer: A Scalable and Elastic Library for Cloud-Native Deep Reinforcement Learning](https://elegantrl.medium.com/elegantrl-podracer-scalable-and-elastic-library-for-cloud-native-deep-reinforcement-learning-bafda6f7fbe0), Dec 11, 2021.
- [Towardsdatascience] [ElegantRL: Mastering PPO Algorithms](https://medium.com/@elegantrl/elegantrl-mastering-the-ppo-algorithm-part-i-9f36bc47b791), May 3, 2021.
- [MLearning.ai] [ElegantRL Demo: Stock Trading Using DDPG (Part II)](https://medium.com/mlearning-ai/elegantrl-demo-stock-trading-using-ddpg-part-ii-d3d97e01999f), Apr 19, 2021.
- [MLearning.ai] [ElegantRL Demo: Stock Trading Using DDPG (Part I)](https://elegantrl.medium.com/elegantrl-demo-stock-trading-using-ddpg-part-i-e77d7dc9d208), Mar 28, 2021.
- [Towardsdatascience] [ElegantRL-Helloworld: A Lightweight and Stable Deep Reinforcement Learning Library](https://towardsdatascience.com/elegantrl-a-lightweight-and-stable-deep-reinforcement-learning-library-95cef5f3460b), Mar 4, 2021.

## ElegantRL-Helloworld

<div align="center">
	<img align="center" src=figs/File_structure.png width="800">
</div>

For beginners, we maintain [ElegantRL-Helloworld](https://github.com/AI4Finance-Foundation/ElegantRL/tree/master/elegantrl_helloworld) as a tutorial. Its goal is to get hands-on experience with ELegantRL.

- Run the [tutorial code and learn about RL algorithms in this order: DQN -> DDPG -> PPO](https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl_helloworld/demo_helloworld_tutorial_DQN_DDPG_PPO.py) or [Jupyter notebook ipynb](https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/tutorial_helloworld_DQN_DDPG_PPO.ipynb)
- Write the [suggestion for Eleagant_HelloWorld in github issue](https://github.com/AI4Finance-Foundation/ElegantRL/issues/135).

One sentence summary: an agent (_agent.py_) with Actor-Critic networks (_net.py_) is trained (_run.py_) by interacting with an environment (_env.py_).

## File Structure

- **elegantrl** # main folder
  - envs, # a collection of environments
  - agent.py, # DRL algorithms
  - config.py, # configurations (hyper-parameter)
  - demo.py, # a collection of demos
  - evaluator.py # the evaluator class
  - net.py # a collection of network architectures
  - replay_buffer.py # the buffer class
  - run.py # training loop
- **elegantrl_helloworld** # tutorial version
  - env.py,
  - agent.py,
  - demo.py,
  - net.py
  - run.py
- **examples** # a collection of example codes
- **ready-to-run Google-Colab notebooks**
  - quickstart_Pendulum_v1.ipynb
  - tutorial_BipedalWalker_v3.ipynb
  - tutorial_Creating_ChasingVecEnv.ipynb
  - tutorial_LunarLanderContinuous_v2.ipynb
- **unit_tests** # a collection of tests

## Experimental Demos

### More efficient than Ray RLlib

Experiments on Ant (MuJoCo), Humainoid (MuJoCo), Ant (Isaac Gym), Humanoid (Isaac Gym) # from left to right

<div align="center">
	<img align="center" src=figs/envs.png width="800">
	<img align="center" src=figs/performance1.png width="800">
	<img align="center" src=figs/performance2.png width="800">
</div>

ElegantRL fully supports Isaac Gym that runs massively parallel simulation (e.g., 4096 sub-envs) on one GPU.

### More stable than Stable-baseline 3

Experiment on Hopper-v2 # ElegantRL achieves much smaller variance (average over 8 runs).

Also, PPO+H in ElegantRL completed the training process of 5M samples about 6x faster than Stable-Baseline3.

<div align="center">
	<img align="center" src=figs/SB3_vs_ElegantRL.png width="640">
</div>

## Testing and Contributing

Our tests are written with the built-in `unittest` Python module for easy access. In order to run a specific test file (for example, `test_training_agents.py`), use the following command from the root directory:

    python -m unittest unit_tests/test_training_agents.py

In order to run all the tests sequentially, you can use the following command:

    python -m unittest discover

Please note that some of the tests require [Isaac Gym](https://developer.nvidia.com/isaac-gym) to be installed on your system. If it is not, any tests related to Isaac Gym will fail.

We welcome any contributions to the codebase, but we ask that you please **do not** submit/push code that breaks the tests. Also, please shy away from modifying the tests just to get your proposed changes to pass them. As it stands, the tests on their own are quite minimal (instantiating environments, training agents for one step, etc.), so if they're breaking, it's almost certainly a problem with your code and not with the tests.

We're actively working on refactoring and trying to make the codebase cleaner and more performant as a whole. If you'd like to help us clean up some code, we'd strongly encourage you to also watch [Uncle Bob's clean coding lessons](https://www.youtube.com/playlist?list=PLmmYSbUCWJ4x1GO839azG_BBw8rkh-zOj) if you haven't already.

## Requirements

    Necessary:
    | Python 3.6+     |
    | PyTorch 1.6+    |

    Not necessary:
    | Numpy 1.18+     | For ReplayBuffer. Numpy will be installed along with PyTorch.
    | gym 0.17.0      | For env. Gym provides tutorial env for DRL training. (env.render() bug in gym==0.18 pyglet==1.6. Change to gym==0.17.0, pyglet==1.5)
    | pybullet 2.7+   | For env. We use PyBullet (free) as an alternative of MuJoCo (not free).
    | box2d-py 2.3.8  | For gym. Use pip install Box2D (instead of box2d-py)
    | matplotlib 3.2  | For plots.

    pip3 install gym==0.17.0 pybullet Box2D matplotlib # or pip install -r requirements.txt
    
    To install StarCraftII env,
    bash ./elegantrl/envs/installsc2.sh
    pip install -r sc2_requirements.txt

## Citation:

To cite this repository:

```
@misc{erl,
  author = {Liu, Xiao-Yang and Li, Zechu and Zheng, Jiahao},
  title = {{ElegantRL}: Massively Parallel Framework for Cloud-native Deep Reinforcement Learning},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AI4Finance-Foundation/ElegantRL}},
}
```
