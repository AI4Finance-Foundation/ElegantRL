# Reinforcement Learning for Multiple-Column Selection

This repository contains the core part of the code for paper: [A Reinforcement-Learning-Based Multiple-Column Selection Strategy for Column Generation](https://arxiv.org/abs/2312.14213) (AAAI2024).

## Introduction

<p align="center">
    <img src="fig/column_generation.png" width= "320">
</p>


We propose the first **reinforcement-learning-based (RL) multiple-column selection strategy** for column generation (CG). The network model contains an encoder, a critic decoder, and an actor decoder, which is trained using the proximal policy optimization (PPO).

<p align="center">
    <img src="fig/neural_network.png" width= "720">
</p>


## Requirements

- python 3.10
- dgl 1.1.1
- gym 0.21.0
- networkx 2.8.7
- numpy 1.23.1
- tianshou 0.4.8
- torch 1.12.1

## Running a Toy Example

A small example data is available in the `state/` folder, where each `.pkl` file corresponds to the state of a column generation iteration for CSP. To run the example, you can directly execute the `example.py`:

```
python example.py
```

We also provide the implementation of our network model in `model.py`, the environment for solving CSP in `env.py` (which should be properly registered in `gym.env`), the basic training process in `training.py` (implemented based on [`tianshou`](https://tianshou.readthedocs.io/en/stable/)), and the evaluation data set in `eval.zip` (please unzip it before training). You can use them as a reference to generate your own data for the problem you want to solve.

## Citation

If you find our work is useful in your research, please consider citing:

```
@inproceedings{Yuan2023ARM,
  title={A Reinforcement-Learning-based Multiple-Column Selection Strategy for Column Generation},
  author={Haofeng Yuan and Lichang Fang and Shiji Song},
  booktitle = {38th AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

## Contact
If you have any questions, please feel free to contact the authors. Haofeng Yuan: yhf22@mails.tsinghua.edu.cn.
