import yaml
from elegantrl_helloworld.agent import AgentDDPG
f = open("hyperparameter.yml", 'r')
a = yaml.safe_load(f)
print(a["DDPG"]["LunarLanderContinuous-v2"])
print(a["DDPG"]["LunarLanderContinuous-v2"]["agent_class"])
agent = a["DDPG"]["LunarLanderContinuous-v2"]["agent_class"]()
print(agent)

