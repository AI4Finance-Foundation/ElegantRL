from .AgentBase import AgentBase

# DQN (off-policy)
from .AgentDQN import AgentDQN, AgentDuelingDQN
from .AgentDQN import AgentDoubleDQN, AgentD3QN
from .AgentEmbedDQN import AgentEmbedDQN, AgentEnsembleDQN

# off-policy
from .AgentTD3 import AgentTD3, AgentDDPG
from .AgentSAC import AgentSAC, AgentModSAC

# on-policy
from .AgentPPO import AgentPPO, AgentDiscretePPO
from .AgentPPO import AgentA2C, AgentDiscreteA2C
