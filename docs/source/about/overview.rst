Key Concepts and Features
=============

One sentence summary: in deep reinforcement learning (DRL), an agent learns by continuously interacting with an unknown environment, in a trial-and-error manner, making sequential decisions under uncertainty and achieving a balance between exploration (of uncharted territory) and exploitation (of current knowledge).

The lifecycle of a DRL application consists of three stages: *simulation*, *learning*, and *deployment*. Our goal is to leverage massive computing power to address three major challenges existed in these three stages: 
  - simulation speed bottleneck;
  - sensitivity to hyper-parameters;
  - unstable generalization ability.

ElegantRL is a massively parallel framework for cloud-native DRL applications implemented in PyTorch:
  - We embrace the accessibility of cloud computing platforms and follow a cloud-native paradigm in the form of containerization, microservices, and orchestration, to ensure fast and robust execution on a cloud.
  - We fully exploit the parallelism of DRL algorithms at multiple levels, namely the worker/learner parallelism within a container, the pipeline parallelism (asynchronous execution) over multiple microservices, and the inherent parallelism of the scheduling task at an orchestrator. 
  - We take advantage of recent technology breakthroughs in massively parallel simulation, population-based training that implicitly searches for optimal hyperparameters, and ensemble methods for variance reduction. 
  
  
**ElegantRL features strong scalability, elasticity and stability and allows practitioners to conduct efficient training from one GPU to hundreds of GPUs on a cloud:**

**Scalable**: the multi-level parallelism results in high scalability. One can train a population with hundreds of agents, where each agent employs thousands of workers and tens of learners. Therefore, ElegantRL can easily scale out to a cloud with hundreds or thousands of nodes.

**Elastic**: ElegantRL features strong elasticity on the cloud. The resource allocation can be made according to the numbers of workers, learners, and agents and the unit resource assigned to each of them. We allow a flexible adaptation to meet the dynamic resource availability on the cloud or the demands of practitioners.

**Stable**: With the massively computing power of a cloud, ensemble methods and population-based training will greatly improve the stability of DRL algorithms. Furthermore, ElegantRL leverages computing resource to implement the Hamiltonian-term as an add-on regularization to model-free DRL algorithms. Such an add-on H-term utilizes computing power (can be computed in parallel on GPU) to search for the "minimum-energy state", corresponding to the stable state of a system. Altogether, ElegantRL demonstrates a much more stable performance compared to Stable-Baseline3, a popular DRL library devote to stability. 

**Accessible**: ElegantRL is a highly modularized framework and maintains ElegantRL-HelloWorld for beginners to get started. We also help users overcome the learning curve by providing API documentations, Colab tutorials, frequently asked questions (FAQs), and demos, e.g., on OpenAI Gym, MuJoCo, Isaac Gym.

  


