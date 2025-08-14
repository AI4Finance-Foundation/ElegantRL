This code is meant to serve as a starting point for future research on _learning to select_ with imitation learning and/or
reinforcement learning. The code has been distributed over 5 steps.

1. Generate instances | (data/{problem}/instances/{instance_type}\_{difficulty}/instance_*.lp)
2. Generate solutions | (data/{problem}/instances/{instance_type}\_{difficulty}/instance_\*-*.sol)  
   (a) Remove infeasible instances and instances with less than 100 nodes explored.
3. Generate samples   | (data/{problem}/samples/{instance_type}\_{difficulty}/sample_*.pkl)  
   (a) For each [instance, solutions] set, generate [state, action] pairs from the oracle in nodesel_oracle.py.  
   (b) The state includes both nodes of the comparison, with a representation defined in extract.py.  
   (c) The action is sampled using the Sampler class, denoted with [0, 1] for left and right respectively.
4. Train model RL/IL  | (experiments/{problem}/04_train_il/{seed}\_{timestamp}/best_params_il_{mode}.pkl)
                      | (actor/{problem}/{model_id}.pkl)  
   (a) MLP policy: [branching_features, node_features, global_features]  
   (b) GNN policy: [not fully implemented]
5. Evaluate policies  | (experiments/{problem}/05_evaluate/{seed}_{timestamp}/{experiment_id}_results.csv)  
   (a) Evaluates on all available [test] and [transfer] instances, with results averaged over 5 runs.  
   (b) Evaluation results are returned using geometric mean and geometric standard deviation.  
   (c) Results can be recalculated based on the csv files using postprocessing.py.

Implemented reward signals: Global tree size, Primal bound improvement, Optimality-Bound difference.
- Global tree size: -1 for each step
- Primal bound improvement: (New GUB - Old GUB) / gap
- Lower bound - objective inequality: -1 if LB > Opt else 0