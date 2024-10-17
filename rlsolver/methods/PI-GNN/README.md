# Combinatorial optimization with physics-inspired graph neural networks

This is my implementation of the [PI-GNN paper](https://arxiv.org/abs/2107.01188), using [`torch_geometric`](https://pytorch-geometric.readthedocs.io/).

## Data

Data here is from [this publicly available datset](https://web.stanford.edu/~yyye/yyye/Gset/), the same as that being used in the numerical benchmark of the original paper. You can download it using the helper script:

```bash
python3 data-helper.py
```

The default directory hierarchy is like below:

```
.
├── data/
│   ├── dataset/
│   ├── links.txt
│   └── raw/
├── log/
├── model/
├── model.py
├── params.py
└── utils.py
```

where:

- `model.py` is the main file for the model.
- `params.py` contains the parameters for the model that one can tweak.
- `utils.py` contains the manipulation of data and a logger class.
- Processed data are stored in `data/dataset/` in the form of `nx.Graph` objects, and the Hamiltonian matrix is only calculated when the data is loaded for the sake of memory efficiency (This could be improved if replaced with a sparse matrix along with sparse multiplication in place of corresponding standard matrix multiplication).

## Environment

For some reason I cannot get `torch` and `torch_geometric` run on Python 3.10, so I am using Python 3.9 instead, with `torch==1.12.0` and `torch_geometric==2.0.4`. Installation instructions can be found [here for Pytorch](https://pytorch.org/get-started/locally/) and [here for PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), or you can use following command to install them:

```bash
# For Pytorch
pip3 install th torchvision torchaudio
# This is for PyG, note that th==1.12.0 is not officialy supported yet.
# Use at your own risk.
pip install th-scatter th-sparse th-cluster th-spline-conv th-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
```