## ElegantRL “小雅”: Scalable and Elastic Deep Reinforcement Learning
 
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


[ElegantRL](https://elegantrl.readthedocs.io/en/latest/index.html) is developed for researchers and practitioners with the following advantages:
  
  + **Lightweight**: the core codes  <1,000 lines (check elegantrl/tutorial), using PyTorch (train), OpenAI Gym (env), NumPy, Matplotlib (plot).
  
  + **Efficient**: in many testing cases, we find it more efficient than [Ray RLlib](https://github.com/ray-project/ray).
  
  + **Stable**: much more stable than [Stable Baselines 3] (https://github.com/DLR-RM/stable-baselines3). Stable Baselines 3 can only use single GPU, but ElegantRL can use 1~8 GPUs for stable training. 

ElegantRL implements the following model-free deep reinforcement learning (DRL) algorithms: 
+ **DDPG, TD3, SAC, PPO, PPO (GAE),REDQ** for continuous actions
+ **DQN, DoubleDQN, D3QN, SAC** for discrete actions
+ **QMIX, VDN; MADDPG, MAPPO, MATD3** for multi-agent environment

For the details of DRL algorithms, please check out the educational webpage [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/). 

《诗经·小雅·鹤鸣》中「他山之石，可以攻玉」，是我们的库“小雅”名字的来源。

## Contents

+ [News](#News)
+ [File Structure](#File-Structure)
+ [Training Pipeline](#Training-Pipeline)
+ [Experimental Results](#Experimental-Demos)
+ [Requirements](#Requirements)
+ [Model-free DRL Algorithms](#Model-free-DRL-Algorithms)

## News

+ [Towardsdatascience] [ElegantRL-Podracer: A Scalable and Elastic Library for Cloud-Native Deep Reinforcement Learning](https://elegantrl.medium.com/elegantrl-podracer-scalable-and-elastic-library-for-cloud-native-deep-reinforcement-learning-bafda6f7fbe0)
+ [Towardsdatascience] [ElegantRL: A Lightweight and Stable Deep Reinforcement Learning Library](https://towardsdatascience.com/elegantrl-a-lightweight-and-stable-deep-reinforcement-learning-library-95cef5f3460b)
+ [Towardsdatascience] [ElegantRL: Mastering PPO Algorithms](https://medium.com/@elegantrl/elegantrl-mastering-the-ppo-algorithm-part-i-9f36bc47b791)
+ [MLearning.ai] [ElegantRL Demo: Stock Trading Using DDPG (Part I)](https://elegantrl.medium.com/elegantrl-demo-stock-trading-using-ddpg-part-i-e77d7dc9d208)
+ [MLearning.ai] [ElegantRL Demo: Stock Trading Using DDPG (Part II)](https://medium.com/mlearning-ai/elegantrl-demo-stock-trading-using-ddpg-part-ii-d3d97e01999f)

## Framework ([Helloworld folder](https://github.com/AI4Finance-Foundation/ElegantRL/tree/master/elegantrl_helloworld))
![File_structure](https://github.com/Yonv1943/ElegantRL/blob/master/figs/File_structure.png)

   An agent (**agent.py**) with Actor-Critic networks (**net.py**) is trained (**run.py**) by interacting with an environment (**env.py**).
   
A high-level overview:
+ 1). Instantiate an environment in **Env.py**, and an agent in **Agent.py** with an Actor network and a Critic network in **Net.py**; 
+ 2). In each training step in **Run.py**, the agent interacts with the environment, generating transitions that are stored into a Replay Buffer; 
+ 3). The agent fetches a batch of transitions from the Replay Buffer to train its networks; 
+ 4). After each update, an evaluator evaluates the agent's performance (e.g., fitness score or cumulative return) and saves the agent if the performance is good.

## Code Structure
### Core Codes
+ **elegantrl/agents/net.py**    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Neural networks.
   + Q-Net,
   + Actor network,
   + Critic network, 
+ **elegantrl/agents/Agent___.py**  &nbsp;&nbsp;# RL algorithms. 
   + AgentBase, 
+ **elegantrl/train/run___.py**    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# run DEMO 1 ~ 4
   + Parameter initialization,
   + Training loop,
   + Evaluator.

### Until Codes
+ **elegantrl/envs/**   &nbsp;&nbsp;&nbsp;&nbsp; # gym env or custom env, including FinanceStockEnv.
   + **gym_utils.py**: A PreprocessEnv class for gym-environment modification.
   + **Stock_Trading_Env**: A self-created stock trading environment as an example for user customization.
+ **eRL_demo_BipedalWalker.ipynb**      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # BipedalWalker-v2 in jupyter notebooks
+ **eRL_demos.ipynb**       &nbsp;&nbsp;&nbsp;&nbsp; # Demo 1~4 in jupyter notebooks. Tell you how to use tutorial version and advanced version.
+ **eRL_demo_SingleFilePPO.py**  &nbsp;&nbsp;&nbsp;&nbsp; # Use a single file to train PPO, more simple than tutorial version
+ **eRL_demo_StockTrading.py**  &nbsp;&nbsp;&nbsp;&nbsp; # Stock Trading Application in jupyter notebooks

## Start to Train

### Initialization:
+ hyper-parameters `args`.
+ `env = PreprocessEnv()` : creates an environment (in the OpenAI gym format).
+ `agent = agent.XXX()` : creates an agent for a DRL algorithm.
+ `buffer = ReplayBuffer()` : stores the transitions.
+ `evaluator = Evaluator()` : evaluates and stores the trained model.

### Training (a while-loop):
+ `agent.explore_env(…)`: the agent explores the environment within target steps, generates transitions, and stores them into the ReplayBuffer.
+ `agent.update_net(…)`: the agent uses a batch from the ReplayBuffer to update the network parameters.
+ `evaluator.evaluate_save(…)`: evaluates the agent's performance and keeps the trained model with the highest score.

The while-loop will terminate when the conditions are met, e.g., achieving a target score, maximum steps, or manually breaks.

## Experiments

## Experimental Demos 

[LunarLanderContinuous-v2](https://gym.openai.com/envs/LunarLanderContinuous-v2/)

![LunarLanderTwinDelay3](https://github.com/Yonv1943/ElegantRL/blob/master/figs/LunarLanderTwinDelay3.gif)

[BipedalWalkerHardcore-v2](https://gym.openai.com/envs/BipedalWalkerHardcore-v2/)

<img src="https://github.com/Yonv1943/ElegantRL/blob/master/figs/BipedalWalkerHardcore-v2-total-668kb.gif" width="150" height="100"/>

Note: BipedalWalkerHardcore is a difficult task in continuous action space. There are only a few RL implementations can reach the target reward. Check out an experiment video: [Crack the BipedalWalkerHardcore-v2 with total reward 310 using IntelAC](https://www.bilibili.com/video/BV1wi4y187tC).

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
  title = {{ElegantRL}: A Scalable and Elastic Deep Reinforcement Learning Library},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AI4Finance-Foundation/ElegantRL}},
}
```


