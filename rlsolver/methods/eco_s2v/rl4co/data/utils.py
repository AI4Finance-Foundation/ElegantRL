import os

import numpy as np
from tensordict.tensordict import TensorDict

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(CURR_DIR))


def load_npz_to_tensordict(filename):
    """Load a npz file directly into a TensorDict
    We assume that the npz file contains a dictionary of numpy arrays
    This is at least an order of magnitude faster than pickle
    """
    x = np.load(filename)
    x_dict = dict(x)
    batch_size = x_dict[list(x_dict.keys())[0]].shape[0]
    return TensorDict(x_dict, batch_size=batch_size)


def save_tensordict_to_npz(tensordict, filename, compress: bool = False):
    """Save a TensorDict to a npz file
    We assume that the TensorDict contains a dictionary of tensors
    """
    x_dict = {k: v.numpy() for k, v in tensordict.items()}
    if compress:
        np.savez_compressed(filename, **x_dict)
    else:
        np.savez(filename, **x_dict)


def check_extension(filename, extension=".npz"):
    """Check that filename has extension, otherwise add it"""
    if os.path.splitext(filename)[1] != extension:
        return filename + extension
    return filename


def load_solomon_instance(name, path=None, edge_weights=False):
    """Load solomon instance from a file"""
    import vrplib

    if not path:
        path = "data/solomon/instances/"
        path = os.path.join(ROOT_PATH, path)
    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = f"{path}{name}.txt"
    if not os.path.isfile(file_path):
        vrplib.download_instance(name=name, path=path)
    return vrplib.read_instance(
        path=file_path,
        instance_format="solomon",
        compute_edge_weights=edge_weights,
    )


def load_solomon_solution(name, path=None):
    """Load solomon solution from a file"""
    import vrplib

    if not path:
        path = "data/solomon/solutions/"
        path = os.path.join(ROOT_PATH, path)
    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = f"{path}{name}.sol"
    if not os.path.isfile(file_path):
        vrplib.download_solution(name=name, path=path)
    return vrplib.read_solution(path=file_path)
