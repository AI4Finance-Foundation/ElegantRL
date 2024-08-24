# RUN-CSP

An optimal graph cut found by a RUN-CSP network for a 25x25 grid graph in 100 iterations:
![](images/animation_compressed.gif)

This repository contains a Tensorflow implementation of RUN-CSP,
a recurrent neural network architecture for Max-CSPs.

We provide all training and evaluation datasets used in our paper.
All instances are stored in the dimacs graph or dimacs cnf formats.
To extract the data, execute the following commands in your shell:

```
cd data
cat *.tar.bz2 | tar jxvf - -i
cd ..
```

We provide all trained network models used in the paper in the 'models' subdirectory.
To train your own models, use the provided training scripts.
For example, our models were trained with the following commands:

```
python3 train_max_2sat.py -m models/2SAT -d data/2SAT_100_Train
python3 train_max_cut.py -m models/Max_Cut -d data/Max_Cut_Train
python3 train_coloring.py -m models/3COL_Pos -d data/3COL_100_Train/positive
python3 train_max_is.py -m models/IS -d data/IS_100_Train
python3 train_max_is.py -m models/IS_RB_Model -d data/RB_Model_Train --kappa 0.1 -b 5
```

The corresponding evaluation scripts can be used to reproduce our experimental results.
For example, the following commands execute the models for Max-2SAT and Max-Cut on the corresponding benchmark instances:

```
python3 evaluate_max_2sat.py -m models/2SAT -d data/Spinglass_2SAT -a 64 -t 100
python3 evaluate_max_cut.py -m models/Max_Cut -d data/GSET -a 64 -t 500
```

To evaluate the models for Max-2SAT and Max-IS on random graphs of a given density/degree (6.0 used as example) use:

```
python3 evaluate_max_2sat.py -m models/2SAT -d data/2SAT_100_Eval/6.0 -a 64 -t 100
python3 evaluate_max_is.py -m models/IS -d data/IS_100_Eval/6.0 -a 64 -t 100
```

To evaluate the networks for 3-COL on hard random instances, use the command:

```python3 evaluate_coloring.py -m models/3COL_Pos_1 -d data/3COL_50_Eval/positive -a 64 -t 100```

An additional script computes the achieved P-values for random regular graphs:

```python3 evaluate_max_cut_regular.py -m models/Max_Cut -d data/Reg_3_500 --degree 3 -a 64 -t 100```

To evaluate a model in the IS benchmark instances of Xu et al., use:

```python3 evaluate_max_is.py -m models/IS_RB_Model -d data/Xu_IS_Benchmarks/frb30-15 -a 8 -t 100```

To execute our greedy Max-IS heuristic on the same graphs use the following command:

```python3 greedy_is.py -d data/RB_Model/frb30-15 ```


Beyond this, we provide a tool to automatically train a RUN-CSP instance for any fixed constraint language.
A Constraint Language is represented as a JSON file that specifies a domain size and the relations.
The model will be trained on randomly generated instance for the specified language.
An example is provided in example_language.json. To train and evaluate a model for this language use:

```
python3 train.py -l example_language.json -m models/example_model
python3 evaluate.py -m models/example_model
```
