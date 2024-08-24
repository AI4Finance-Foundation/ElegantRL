# graph_comb_opt 
Implementation of "Learning Combinatorial Optimization Algorithms over Graphs" (https://arxiv.org/abs/1704.01665)

Step-by-step demo of MVC solution found by different methods.
From left to right: (1) S2V-DQN (our method), (2) node-degree heuristic, (3) edge-degree heuristic
![demo](https://github.com/Hanjun-Dai/graph_comb_opt/blob/master/visualize/mvc-40-50.gif)

# 1. build

**** Below shows an example of MVC. For all the problems, you can follow the similar pipeline ****

Get the source code, and install all the dependencies. 

    git clone --recursive https://github.com/Hanjun-Dai/graph_comb_opt
    
    build the graphnn library with the instructions here:
      https://github.com/Hanjun-Dai/graphnn
    
For each task, build the dynamic library. For example, to build the Minimum Vertex Cover library:

    cd code/s2v_mvc/mvc_lib/
    cp Makefile.example Makefile
    
    customize your Makefile if necessary
    ( add CXXFLAGS += -DGPU_MODE in the Makefile if you want to run it in GPU mode)
    
    make -j
    
Now you are all set with the C++ backend. 

# 2. Experiments on synthetic data

### Generate synthetic data

To generate the synthetic data for Minimum Vertex Cover task, you can do the following:

    cd code/data_generator/mvc
    modify parameters in run_generate.sh
    ./run_generate.sh
    
The above code will generate 1000 test graphs under /path/to/the/project/data/mvc 

### Training with n-step Q-learning

Navigate to the MVC folder and run the training script. Modify the script to change the parameters. 

    cd code/s2v_mvc
    ./run_nstep_dqn.sh
    
By default it will save all the model files, the logs under currentfolder/results. Note that the code will generate the data on the fly, including the validation dataset. So the training code itself doesn't rely on the data generator. 

### Test the performance

Navigate to the MVC folder and run the evaluation script. Modify the script to change the parameters. Make sure the parameters are consistent with your training script. 

    cd code/s2v_mvc
    ./run_eval.sh

The above script will load the 1000 test graphs you generated before, and output the solution in a csv file, under the same results folder. Format of the csv for MVC:

    cover size, cover_size a_1 a_2 a_3 ...., time in seconds
    
    Here the second column shows a solution found by S2V-DQN, in the same order of how each node is picked. 

# 2. Experiments on real-world data

For TSP we test on part of the tsplib instances;
For MVC and SCP, we use memetracker dataset; 
For MAXCUT, we test on optsicom dataset; 

All the data can be found through the dropbox link below. Code folders that start with 'realworld' are for this set of experiments. 

# Reproducing the results that reported in the paper

Here is the link to the dataset that was used in the paper:

https://www.dropbox.com/sh/r39596h8e26nhsp/AADRm5mb82xn7h3BB4KXgETsa?dl=0


# Reference

Please cite our work if you find our code/paper is useful to your work. 

    @article{dai2017learning,
      title={Learning Combinatorial Optimization Algorithms over Graphs},
      author={Dai, Hanjun and Khalil, Elias B and Zhang, Yuyu and Dilkina, Bistra and Song, Le},
      journal={arXiv preprint arXiv:1704.01665},
      year={2017}
    }
