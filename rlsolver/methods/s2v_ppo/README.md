# S2V-PPO for MaxCut

A PyTorch-based implementation of a PPO (Proximal Policy Optimization) reinforcement learning agent for solving the Maximum Cut (MaxCut) problem on graphs. It uses PyTorch Geometric for graph handling, distributed data parallel (DDP) training for multi-GPU support, and a custom environment with tabu search and basin-hopping inspired mechanisms.

## Requirements
- Python 3.8+
- PyTorch 2.0+ (with CUDA for GPU support)
- PyTorch Geometric
- NetworkX
- NumPy
- tqdm
- torch-scatter (installed via PyTorch Geometric)

## Usage

### 1. Generate Data
The code includes a script to generate synthetic graphs for training. Graphs are saved in `./maxcut_data` as `.txt` files (format: first line `n m`, followed by `u v w` for each edge).

Run:
```
python data_generate.py
```
- This generates graphs of sizes [20, 30, 50, 100, 200] with 10 samples each for types: Barabási–Albert (BA), Erdős–Rényi (ER), and Powerlaw Cluster (PL).
- Weighted edges (uniform [1,10]).
- Customize sizes, types, or samples in `data_generate.py` (e.g., edit `sizes` list or `samples_per_size`).

For custom data, place graphs in `./maxcut_data` following the same format.

### 2. Start Training
Training uses DDP for multi-GPU. It loads graphs from `./maxcut_data`, trains the PPO agent, and saves the model as `model.pth`.

Run:
```
python launch.py
```
- Detects available GPUs automatically (requires at least 1 CUDA device).
- Hyperparameters are configurable in `config.py` (e.g., `epochs=500`, `batch_size=8192`, `lr=2e-4`, `hidden_dim=128`, `gamma=0.99`).



### 3. Inference
Evaluation loads the trained model (`model.pth`) and runs greedy inference on test graphs. Input graphs are from `../../data` (relative to script dir), outputs assignments to `../../result` (format: `node_id group` where group is 1 or 2).

Run:
```
python Inference.py
```
- Processes all `.txt` files in `../../data` (skipping `dataset_info.txt`).
- For each graph, outputs the best cut value and node assignments (z=-1 -> 1, z=1 -> 2).
- Customize paths in `Inference.py` if needed (e.g., `data_root` and `result_root`).

To inference on custom graphs, place them in `../../data` or adjust paths.

## Notes
- Training assumes graphs in `./maxcut_data`; evaluation uses a relative benchmark path (adapt for your setup).
