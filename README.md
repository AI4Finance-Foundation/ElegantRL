## Lightweight, Efficient and Stable DRL Implementation Using PyTorch 

<br/>
<a href="https://github.com/AI4Finance-LLC/ElegantRL" target="\_blank">
	<div align="center">
		<img src="Result/ElegantRL.jpg" width="40%"/>
	</div>
<!-- 	<div align="center"><caption>Slack Invitation Link</caption></div> -->
</a>
<br/>

We maintain an **elegant (lightweight, efficient and stable)** lib, for researchers and practitioners.
  
  + **Lightweight**: The core codes have less than 800 code lines, using PyTorch, OpenAI Gym, and NumPy.
  
  + **Efficient**: Its performance is comparable with Ray RLlib [link](https://github.com/ray-project/ray).
  
  + **Stable**: It is as stable as Stable Baseline 3 [link](https://github.com/DLR-RM/stable-baselines3).

Model-free deep reinforcement learning (DRL) Algorithms: 
+ **DDPG, A2C, PPO(GAE), SAC, TD3, InterAC, InterSAC for continuous actions**
+ **DQN, DoubleDQN, DuelingDQN, GAE for discrete actions**

For the algorithms, please check out the [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/). 

### Model-free DRL Algorithms

![model-free_DRL_2020](https://github.com/Yonv1943/ElegantRL/blob/master/Result/model-free_DRL_2020.png)

More policy gradient algorithms (Actor-Critic methods): [Policy gradient algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)


# Experimental results

Results using [ElegantRL](https://github.com/Yonv1943/ElegantRL/blob/master/Result): gif/png ↓ in "/Result".

[LunarLanderContinuous-v2](https://gym.openai.com/envs/LunarLanderContinuous-v2/)

![LunarLanderTwinDelay3](https://github.com/Yonv1943/ElegantRL/blob/master/Result/LunarLanderTwinDelay3.gif)

[BipedalWalkerHardcore-v2](https://gym.openai.com/envs/BipedalWalkerHardcore-v2/)


<img src="https://github.com/Yonv1943/ElegantRL/blob/master/Result/BipedalWalkerHardcore-v2-total-668kb.gif" width="150" height="150"/>


BipedalWalkerHardcore is a difficult task in continuous action space. There are only a few RL implementations can reach the target reward.
Check out our video on bilibili: [Crack the BipedalWalkerHardcore-v2 with total reward 310 using IntelAC](https://www.bilibili.com/video/BV1wi4y187tC).

For example, in the following two figures.

+ The first is a luck result of BasicAC: `plot_0072E_22405T_701s.png`.
+ The second is a slow result of BasicAC: `plot_0249E_159736T_1867s.png`. 
+ The name of the log figure is `plot_<Episode>_<TotalStep>_<UsedTimeInSecond>.png` when agent reaches the default target reward.

BasicAC is an improved DDPG. SAC and DeepSAC are more stable.

<p float="left">
  <img src="https://github.com/Yonv1943/ElegantRL/blob/master/Result/BasicAC_LunarLanderContinuous-v2_luck/plot_0072E_22405T_701s.png" width="400" />
  <img src="https://github.com/Yonv1943/ElegantRL/blob/master/Result/BasicAC_LunarLanderContinuous-v2_unluck/plot_0249E_159736T_1867s.png" width="400" /> 
</p>

In the above two figures. 
+ Blue curve: Total reward (with exploration noise)
+ Red curve: Total reward (without exploration noise, mean and std)
+ Grey area: The step of one episode while exploring.
+ Green curve: 'loss' value of actor (mean of Q value estimate)
+ Green area: loss value of critic

InterSAC trains 1e5s (28hours) in 2.6e6 steps, and get 310 reward, which is quite fast and good.
![InterSAC_BipedalWalkerHardcore](https://github.com/Yonv1943/ElegantRL/blob/master/Result/InterSAC_BipedalWalkerHardcore-v3_310/plot_Step_Time_2665512_102194.png)


More training logs in "Result". 


# Requirement

    Necessary:
    | Python 3.7           
    | PyTorch 1.0.2       

    Not necessary:
    | Numpy 1.19.0   
    | gym 0.17.2             
    | box2d-py 2.3.8    
    | matplotlib 3.2  | For plots
It is **lightweight**.

# File Structure
    -----file----
    AgentRun.py  # Choose an RL agent here.
    AgentZoo.py  # Model-free RL algorithms.
    AgentNet.py  # Neural networks. 
    Tutorial.py  # A turorial: simplify DQN and DDPG are here.
    
    ----folder---
    BetaWarning  # The latest version, may inclue bugs.
    History      # Historical files here.
There are only 3 python files. (`Tutorial.py` is **not** a part of the lib.)

# Run
    python3 AgentRun.py
    # You can see run__demo(gpu_id=0, cwd='AC_BasicAC') in AgentRun.py.
+ In default, it will train a stable-DDPG in LunarLanderContinuous-v2 for 2000 second.
+ It would choose CPU or GPU automatically. Don't worry, I never use `.cuda()`.
+ It would save the log and model parameters file in Current Working Directory `cwd='AC_BasicAC'`. 
+ It would print the total reward while training. Maybe I should use TensorBoardX?
+ There are many comment in the code. I believe these comments can answer some of your questions.

### Use other DRL algorithms?
The following steps:
1. See `run__xxx()` in `AgentRun.py`.
2. Use `run__zoo()` to run an off-policy algorithm. Use `run__ppo()` to run on-policy such as PPO.
3. Choose a DRL algorithm: `from AgentZoo import AgentXXX`.
4. Choose a gym environment: `args.env_name = "LunarLanderContinuous-v2"`
