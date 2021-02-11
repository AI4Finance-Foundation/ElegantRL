## Lightweight, Efficient and Stable DRL Implementation Using PyTorch 

Model-free deep reinforcement learning (DRL) Algorithms: 
+ **DDPG, A2C, PPO(GAE), SAC, TD3, InterAC, InterSAC for continuous actions**
+ **DQN, DoubleDQN, DuelingDQN, GAE for discrete actions**

For the algorithms, please check out the [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/). 

# Experimental results

You can see these gif/png ↓ in file "/Result" in [ElegantRL](https://github.com/Yonv1943/ElegantRL/blob/master/Result).

LunarLanderContinuous-v2:
![LunarLanderTwinDelay3](https://github.com/Yonv1943/ElegantRL/blob/master/Result/LunarLanderTwinDelay3.gif)
BipedalWalkerHardcore-v2: 
![BipedalWalkerHardcore-v2-total](https://github.com/Yonv1943/ElegantRL/blob/master/Result/BipedalWalkerHardcore-v2-total-668kb.gif)

BipedalWalkerHardcore is a hard task in a continuous action space. There is less RL implementations that can reach the target reward.
You can also check the video [bilibili Faster Pass BipedalWalkerHardcore-v2 total reward 310](https://www.bilibili.com/video/BV1wi4y187tC).

For example, the training log is the following two figures.

+ The first is a luck result of BasicAC: `plot_0072E_22405T_701s.png`.
+ The second is a slow result of BasicAC: `plot_0249E_159736T_1867s.png`. 
+ The name of the log figure is `plot_<Episode>_<TotalStep>_<UsedTimeInSecond>.png` when agent reaches the default target reward.

BasicAC is an improved DDPG, not in a stable status at this moment. You can try other DRL algorithms, such as SAC and DeepSAC, which are more stable than other algorithms.

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

BipedalWalkerHardcore is a hard task in a continuous action space. There is less RL implementation can reach the target reward. 
I am happy that InterSAC trains 1e5s (28hours) in 2.6e6 steps, and get 310 reward. It is quite fast and good.
![InterSAC_BipedalWalkerHardcore](https://github.com/Yonv1943/ElegantRL/blob/master/Result/InterSAC_BipedalWalkerHardcore-v3_310/plot_Step_Time_2665512_102194.png)


You can see more training log in file "Result". 


# Requirement

    Necessary:
    | Python 3.7      | Python 3.5+ is ok, because I use Python's multiprocessing     
    | PyTorch 1.0.2   | PyTorch 1.X is ok. After all, PyTorch is not TensorFlow.      

    Not necessary:
    | Numpy 1.19.0    | Numpy 1.X is ok. Numpy is one of the requirement for PyTorch. 
    | gym 0.17.2      | In early gym version, some task XXX-v3 would be XXX-v2        
    | box2d-py 2.3.8  | This box2d-py is OpenAI's instead of the original author's    
    | matplotlib 3.2  | Just for drawing plot
This is why it is **lightweight**.

# File Structure
    -----file----
    AgentRun.py  # Choose your RL agent here. Then run it for training.
    AgentZoo.py  # Many model-free RL algorithms are here.
    AgentNet.py  # The neural network architectures are here. 
    Tutorial.py  # It is a turorial for RL learner. Simplify DQN and DDPG are here.
    
    ----folder---
    BetaWarning  # I put the latest version here. You can find lots of bug here, but new.
    History      # I put the historical files here. You can find the bug-free code here, but old.
There are only 3 python file in this DRL PyTorch implementation. (`Tutorial.py` is **not** a part of the lib.)

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

### Model-free DRL Algorithms

![model-free_DRL_2020](https://github.com/Yonv1943/ElegantRL/blob/master/Result/model-free_DRL_2020.png)

You can find above figure in `./Temp/model-free_DRL_2020.png` or `*.pdf`.

More Policy Gradient Algorithms (Actor-Critic Methods):
+ [Policy gradient algorithms by Lilian Weng](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)


#### Plan to add 
- add FlappyBird-v0 (PyGame), 2D state and 1D state

Soft Actor-Critic for Discrete Action Settings https://www.arxiv-vanity.com/papers/1910.07207/

Multi-Agent Deep RL: MADDPG, QMIX, QTRAN

some variants of DQN: Rainbow DQN, Ape-X.
