from elegantrl.agents.net import QNetDuel
from elegantrl.agents.AgentDQN import AgentDQN


class AgentDuelingDQN(AgentDQN):  # [ElegantRL.2022.04.04]
    """
    Dueling network.
    """

    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.act_class = getattr(self, "act_class", QNetDuel)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        AgentDQN.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)
