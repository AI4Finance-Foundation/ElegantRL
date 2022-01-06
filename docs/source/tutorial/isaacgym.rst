How to run worker parallelism: Isaac Gym
======================================================

For GPU-accelerated VecEnv, ElegantRL supports both external GPU-accelerated VecEnv, e.g., NVIDIA Isaac Gym, and user-customized VecEnv. Next, we explain in detail how to use Isaac Gym and how to define your own VecEnv in ElegantRL.

Running a VecEnv environment from NVIDIA Isaac Gym
------------------------------------------

Isaac Gym is NVIDIA’s prototype physics simulators for reinforcement learning research. Isaac Gym includes typical RL tasks, e.g., Cartpole, Ant, Humanoid, Shadow Hand Object Manipulation, and also supports user-customization. Please follow the instructions at https://developer.nvidia.com/isaac-gym.

ElegantRL provides a wrapper ``PreprocessIsaacVecEnv`` to process an Isaac Gym environment:

.. code-block:: python

    from elegantrl.envs.IsaacGym import PreprocessIsaacVecEnv

    env_name = 'Ant'

    env = PreprocessIsaacVecEnv(env_name, if_print=False, env_num=4096, device_id=0)


