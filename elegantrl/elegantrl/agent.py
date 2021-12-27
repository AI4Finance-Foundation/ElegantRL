from elegantrl.agents.AgentBase import AgentBase

from elegantrl.agents.AgentDQN import AgentDQN, AgentDuelingDQN
from elegantrl.agents.AgentDoubleDQN import AgentDoubleDQN, AgentD3QN

from elegantrl.agents.AgentA2C import AgentA2C, AgentShareA2C, AgentDiscreteA2C
from elegantrl.agents.AgentPPO import AgentPPO, AgentSharePPO, AgentDiscretePPO

from elegantrl.agents.AgentDDPG import AgentDDPG
from elegantrl.agents.AgentTD3 import AgentTD3
from elegantrl.agents.AgentSAC import AgentSAC, AgentModSAC, AgentShareSAC
from elegantrl.agents.AgentREDQ import AgentREDQ
from elegantrl.agents.AgentStep1AC import AgentStep1AC, AgentShareStep1AC

from elegantrl.agents.AgentQMix import AgentQMix
from elegantrl.agents.AgentVDN import AgentVDN
from elegantrl.agents.AgentMADDPG import AgentMADDPG
from elegantrl.agents.AgentMATD3 import AgentMATD3
from elegantrl.agents.AgentMAPPO import AgentMAPPO

dir((
    AgentBase,

    AgentDQN, AgentDuelingDQN,  # DQN
    AgentDoubleDQN, AgentD3QN,  # DoubleDQN

    AgentA2C, AgentShareA2C, AgentDiscreteA2C,  # on-policy A2C
    AgentPPO, AgentSharePPO, AgentDiscretePPO,  # on-policy PPO

    AgentDDPG, AgentTD3, AgentSAC, AgentModSAC, AgentShareSAC,  # off-policy: DDPG-style
    AgentREDQ, AgentStep1AC, AgentShareStep1AC,  # off-policy: other

    AgentQMix, AgentVDN,  # MARL: QMix
    AgentMADDPG, AgentMATD3, AgentMAPPO  # MARL: CTDE
))
