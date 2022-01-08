Features
=============

One sentence summary: in deep reinforcement learning (DRL), an agent learns by continuously interacting with an unknown environment, in a trial-and-error manner, making sequential decisions under uncertainty and achieving a balance between exploration (of uncharted territory) and exploitation (of current knowledge).

Since its resurgence in 2013, DRL has revolutionized learning and actuation in game playing and robot control. However, it may take months to adapt DRL methods to real-world applications, e.g., recommendation systems, autonomous driving, algorithmic trading, and computer networks. A major challenge is to balance the exploration and exploitation during the training process, which are compute-intensive and time-consuming. 

ElegantRL is an open-source massively parallel framework for DRL algorithms implemented in PyTorch. 

  - We fully exploit the parallelism of DRL algorithms at multiple levels, including agent parallelism of population-based training and worker-learner parallelism of a single agent. 

  - We emphasize the importance of ensemble methods, e.g., adaptive parallelism and weighted average, which perform remarkably well in practice. 

  - We follow the cloud-native paradigm, implement the training process as a synergy of microservices, and achieve containerization, ensuring fast and robust execution on cloud platforms. 

**Scalable**: the multi-level parallelism results in high scalability. E.g., we can train hundreds of DRL agents, where each agent is allocated with fixed computing resources, and perform a tournament-based evolution among the agents via adaptive scheduling. The training processes of agents are asynchronous and distributed, eliminating the agent-to-agent communication. In this way, ElegantRL can easily scale out to hundreds or thousands of computing nodes on a cloud platform, say, thousands of GPUs.

**Elastic**: ElegantRL can elastically allocate computing resources by adjusting the number of agents. We provide an orchestrator to monitor the current training status and the available computing resources, enabling the dynamic resource management on a cloud. For example, when the resources are limited, the orchestrator can automatically kill agents with low performance, which helps the application adapt to available resources and prevents over-provisioning and under-provisioning.

**Flexible**: ElegantRL allows a flexible combination of the number of workers, learners, and agents, with only a few lines of change in the configuration. Such flexibility accommodates the demands of users with different training purposes. For example,

  - To accelerate data collection for complex simulations, e.g., robotic control, users can use multiple workers to generate transitions in parallel. 

  - To improve the sample efficiency and/or stabilize the learning process, users can use multiple learners to update network parameters in parallel. 

  - To employ ensemble methods on the cloud, users can train multiple agents in parallel that are initialized with different DRL algorithms and hyper-parameters, e.g., population-based training and tournament-based ensemble training.

**Efficient**: ElegantRL can efficiently train a powerful DRL agent by scheduling the multi-level parallelism and ensemble methods onto GPU devices. For Ant and Humanoid tasks, we demonstrate that ElegantRL outperforms popular libraries on one DGX-2 server, e.g., a 5~10x speed-up in training time over RLlib. For stock trading tasks, we successfully obtain a profitable trading agent in 10 minutes on an NVIDIA DGX SuperPOD cloud with 80 A100 GPUs, for a stock trading task on NASDAQ-100 constituent stocks with minute-level data over 5 years.

**Accessible**: ElegantRL connects microservices via simple but clearly defined API, allowing users to safely use and customize the framework without understanding its every detail. In addition, we exploit the hierarchical development of DRL algorithms (e.g., from DDPG, TD3 to SAC) to reduce code duplication, making our implementations lightweight. We also help users overcome the learning curve by providing API documentations, Colab tutorials, frequently asked questions (FAQs), and demos, e.g., on OpenAI Gym, MuJoCo, Isaac Gym.

  


