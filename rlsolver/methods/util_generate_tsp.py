import os
import numpy as np


class ConfigTsp:
    batch_size = 1
    num_nodes = 10
    low = 0
    high = 100
    random_mode = 'uniform'  # 'uniform','gaussian'
    assert random_mode in ['uniform', 'gaussian']
    filename = f"tsp{num_nodes}_batch{batch_size}_{random_mode}"
    data_path = "../data/" + filename + '.tsp'


def generate_tsp_data(batch=10, nodes_num=10, low=0, high=1, random_mode="uniform"):
    if random_mode == "uniform":
        node_coords = np.random.uniform(low, high, size=(batch, nodes_num, 2))
    elif random_mode == "gaussian":
        node_coords = np.random.normal(loc=0, scale=1, size=(batch, nodes_num, 2))
        max_value = np.max(node_coords)
        min_value = np.min(node_coords)
        node_coords = np.interp(node_coords, (min_value, max_value), (low, high))
    else:
        raise ValueError(f"Unknown random_mode: {random_mode}")
    return node_coords


def generate_tsp_file(node_coords: np.ndarray, filename):
    if node_coords.ndim == 3:
        shape = node_coords.shape
        if shape[0] == 1:
            node_coords = node_coords.squeeze(axis=0)
            _generate_tsp_file(node_coords, filename)
        else:
            for i in range(shape[0]):
                _filename = filename.replace('.tsp', '') + '_ID' + str(i) + '.tsp'
                _generate_tsp_file(node_coords[i], _filename)
    else:
        assert node_coords.ndim == 2
        _generate_tsp_file(node_coords, filename)


def _generate_tsp_file(node_coords: np.ndarray, filename):
    num_points = node_coords.shape[0]
    file_basename = os.path.basename(filename)
    with open(filename, 'w') as f:
        f.write(f"NAME: {file_basename}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {num_points}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i in range(num_points):
            x, y = node_coords[i]
            f.write(f"{i + 1} {x} {y}\n")
        f.write("EOF\n")


if __name__ == "__main__":
    # tab_printer(args)
    node_coords = generate_tsp_data(ConfigTsp.batch_size, ConfigTsp.num_nodes, ConfigTsp.low, ConfigTsp.high, ConfigTsp.random_mode)
    generate_tsp_file(node_coords, ConfigTsp.data_path)
