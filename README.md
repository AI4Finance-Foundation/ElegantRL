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


[ElegantRL](https://elegantrl.readthedocs.io/en/latest/index.html) is developed for practitioners with
the following advantages:

+ **Cloud-native**: follows a cloud-native paradigm through microservice architecture and containerization, supporting [ElegantRL-Podracer](https://elegantrl.readthedocs.io/en/latest/tutorial/elegantrl-podracer.html) and [FinRL-Podracer](https://elegantrl.readthedocs.io/en/latest/tutorial/finrl-podracer.html).

+ **Scalable**: fully exploits the parallelism of DRL algorithms at multiple levels, making it easily scale out to hundreds or thousands of computing nodes on a cloud platform, say, a [DGX SuperPOD platform](https://www.nvidia.com/en-us/data-center/dgx-superpod/) with thousands of GPUs.

+ **Elastic**: allows to elastically and automatically allocate computing resources on the cloud.

+ **Lightweight**: the core codes  <1,000 lines (check [Elegantrl_Helloworld](https://github.com/AI4Finance-Foundation/ElegantRL/tree/master/elegantrl_helloworld)).

+ **Efficient**: in many testing cases (single GPU/multi-GPU/GPU cloud), we find it more efficient than [Ray RLlib](https://github.com/ray-project/ray).

+ **Stable**: much much much more stable than [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) by utilizing various ensemble methods.

ElegantRL implements the following model-free deep reinforcement learning (DRL) algorithms:

+ **DDPG, TD3, SAC, PPO, REDQ** for continuous actions,
+ **DQN, Double DQN, D3QN, SAC** for discrete actions,
+ **QMIX, VDN, MADDPG, MAPPO, MATD3** for multi-agent environment.

For the details of DRL algorithms, please check out the educational
webpage [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/).

ElegantRL supports the following simulators:

+ **Isaac Gym** for massively parallel simulation,
+ **OpenAI Gym, MuJoCo, PyBullet, FinRL** for benchmarking.

“小雅”源于《诗经·小雅·鹤鸣》，旨在「他山之石，可以攻玉」。

## Contents

+ [News](#News)
+ [ElegantRL-Helloworld](#ElegantRL-Helloworld)
+ [File Structure](#File-Structure)
+ [Experimental Demos](#Experimental-Demos)
+ [Requirements](#Requirements)
+ [Citation](#Citation)

## News

+ [Towardsdatascience] [ElegantRL-Podracer: A Scalable and Elastic Library for Cloud-Native Deep Reinforcement Learning](https://elegantrl.medium.com/elegantrl-podracer-scalable-and-elastic-library-for-cloud-native-deep-reinforcement-learning-bafda6f7fbe0), Dec 11, 2021.
+ [Towardsdatascience] [ElegantRL: Mastering PPO Algorithms](https://medium.com/@elegantrl/elegantrl-mastering-the-ppo-algorithm-part-i-9f36bc47b791), May 3, 2021.
+ [MLearning.ai] [ElegantRL Demo: Stock Trading Using DDPG (Part II)](https://medium.com/mlearning-ai/elegantrl-demo-stock-trading-using-ddpg-part-ii-d3d97e01999f), Apr 19, 2021.
+ [MLearning.ai] [ElegantRL Demo: Stock Trading Using DDPG (Part I)](https://elegantrl.medium.com/elegantrl-demo-stock-trading-using-ddpg-part-i-e77d7dc9d208), Mar 28, 2021.
+ [Towardsdatascience] [ElegantRL-Helloworld: A Lightweight and Stable Deep Reinforcement Learning Library](https://towardsdatascience.com/elegantrl-a-lightweight-and-stable-deep-reinforcement-learning-library-95cef5f3460b), Mar 4, 2021.


## ElegantRL-Helloworld

<div align="center">
	<img align="center" src=figs/File_structure.png width="800">
</div>

For beginners, we maintain [ElegantRL-Helloworld](https://github.com/AI4Finance-Foundation/ElegantRL/tree/master/elegantrl_helloworld) as a tutorial. Its goal is to get hands-on experience with ELegantRL.

One sentence summary: an agent (*agent.py*) with Actor-Critic networks (*net.py*) is trained (*run.py*) by interacting with an environment (*env.py*).

## File Structure

+ **elegantrl**		# main folder
    + envs,        	# a collection of environments   
    + agent.py,    	# DRL algorithms
    + config.py,   	# configurations (hyper-parameter)
    + demo.py, 	   	# a collection of demos
    + evaluator.py 	# the evaluator class
    + net.py	  	# a collection of network architectures
    + replay_buffer.py	# the buffer class
    + run.py	   	# training loop
+ **elegantrl_helloworld**  # tutorial version
    + env.py,        	
    + agent.py,    	
    + demo.py, 	 
    + net.py	  	
    + run.py	   	
+ **examples**		# a collection of example codes
+ **ready-to-run Google-Colab notebooks**
    + quickstart_Pendulum_v1.ipynb
    + tutorial_BipedalWalker_v3.ipynb
    + tutorial_Creating_ChasingVecEnv.ipynb
    + tutorial_LunarLanderContinuous_v2.ipynb 

## Experimental Demos

### More efficient than Ray RLlib

Experiments on Ant (MuJoCo), Humainoid (MuJoCo), Ant (Isaac Gym), Humanoid (Isaac Gym) # from left to right

<div align="center">
	<img align="center" src=figs/envs.png width="800">
	<img align="center" src=figs/performance.png width="800">
</div>

ElegantRL fully supports Isaac Gym that runs massively parallel simulation (e.g., 4096 sub-envs) on one GPU.

### More stable than Stable-baseline 3

Experiment on Hopper-v2 # ElegantRL achieves higher average rewards and much smaller variance (average over 10 runs)

<div align="center">
	<img align="center" src=figs/SB3_vs_ElegantRL.png width="640">
</div>


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
    
    pip3 install gym==0.17.0 pybullet Box2D matplotlib

    To install StarCraftII env,
    bash ./elegantrl/envs/installsc2.sh
    pip install -r sc2_requirements.txt

## Citation:

To cite this repository:

```
@misc{erl,
  author = {Liu, Xiao-Yang and Li, Zechu and Wang, Zhaoran and Zheng, Jiahao},
  title = {{ElegantRL}: Massively Parallel Framework for Cloud-native Deep Reinforcement Learning},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AI4Finance-Foundation/ElegantRL}},
}
```


