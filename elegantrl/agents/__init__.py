from elegantrl.agents.AgentBase import AgentBase

# DQN (off-policy)
from elegantrl.agents.AgentDQN import AgentDQN, AgentDuelingDQN
from elegantrl.agents.AgentDQN import AgentDoubleDQN, AgentD3QN

# off-policy
from elegantrl.agents.AgentDDPG import AgentDDPG
from elegantrl.agents.AgentTD3 import AgentTD3
from elegantrl.agents.AgentSAC import AgentSAC, AgentModSAC

# on-policy
from elegantrl.agents.AgentPPO import AgentPPO, AgentDiscretePPO
from elegantrl.agents.AgentA2C import AgentA2C, AgentDiscreteA2C
