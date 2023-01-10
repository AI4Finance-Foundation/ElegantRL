from elegantrl.agents.base import AgentBase

# DQN (off-policy)
from elegantrl.agents.dqn import AgentDQN, AgentDuelingDQN
from elegantrl.agents.dqn import AgentDoubleDQN, AgentD3QN

# off-policy
from elegantrl.agents.ddpg import AgentDDPG
from elegantrl.agents.td3 import AgentTD3
from elegantrl.agents.sac import AgentSAC, AgentModSAC

# on-policy
from elegantrl.agents.ppo import AgentPPO, AgentDiscretePPO
from elegantrl.agents.a2c import AgentA2C, AgentDiscreteA2C
