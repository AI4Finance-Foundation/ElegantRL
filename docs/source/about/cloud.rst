Cloud-native Paradigm
=================================
To the best of our knowledge, ElegantRL is the first open-source cloud-native framework that supports millions of GPU cores to carry out massively parallel DRL training at multiple levels.

In this article, we will discuss our motivation and cloud-native designs.

Why cloud-native?
---------------------------------

When you need more computing power and storage for your task, running on a cloud may be a more preferable choice than buying racks of machines. Due to its accessible and automated nature, the cloud has been a disruptive force in many deep learning tasks, such as natural langauge processing, image recognition, video synthesis, etc.

Therefore, we embrace the cloud computing platforms to:

  - build a serverless application framework that performs the entire life-cycle (simulate-learn-deploy) of DRL applications on low-cost cloud computing power.
  - support for single-click training for sophisticated DRL problems (compute-intensive and time-consuming) with automatic hyper-parameter tuning.
  - provide off-the-shelf APIs to free users from full-stack development and machine learning implementations, e.g., DRL algorithms, ensemble methods, performance analysis. 

Our goal is to allow for wider DRL applications and faster development life cycles that can be created by smaller teams. One simple example of this is the following workflow.

A user wants to train a trading agent using minute-level NASDAQ 100 constituent stock dataset, a compute-intensive task as the dimensions of the dataset increase, e.g., the number of stocks, the length of period, the number of features. Once the user finishes constructing the environment/simulator, she can directly submit the job to our framework. Say the user has no idea which DRL algorithms she should use and how to setup the hyper-parameters, the framework can automatically initialize agents with different algorithms and hyper-parameter to search the best combination. All data is stored in the cloud storage and the computing is parallized on cloud clusters.

A cloud-native solution
-----------------------------------------------------------------------

ElegantRL follows the cloud-native paradigm in the form of microservice, containerization, and orchestration.

**Microservices**: ElegantRL organizes a DRL agent as a collection of microservices, including orchestrator, worker, learner, evaluator, etc. Each microservice has specialized functionality and connects to other microservices through clear-cut APIs. The microservice structure makes ElegantRL a highly modularized framework and allows practitioners to use and customize without understanding its every detail.

**Containerization**: An agent is encapsulated into a pod (the  basic deployable object in Kubernetes (K8s)), while each microservice within the agent is mapped to a container (a lightweight and portable package of software). On the cloud, microservice and containerization together offer significant benefits in asynchronous parallelism, fault isolation, and security.

**Orchestration**: ElegantRL employs K8s to orchestrate pods and containers, which automates the deployment and management of the DRL application on the cloud. Our goal is to free developers and practitioners from sophisticated distributed machine learning.

We provide two different scheduling mechanism on the cloud, namely generational evolution and tournament-based evolution. 

A tutorial on generational evolution is available `here <https://elegantrl.readthedocs.io/en/latest/tutorial/finrl-podracer.html>`_.

A tutorial on tournament-based evolution is available `here <https://elegantrl.readthedocs.io/en/latest/tutorial/elegantrl-podracer.html>`_.
