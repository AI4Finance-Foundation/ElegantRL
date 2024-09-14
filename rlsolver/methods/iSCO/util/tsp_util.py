from rlsolver.methods.iSCO.config.tsp_config import *
import os
import torch
import numpy as np

def load_data(tsp_file_path):
    data = torch.tensor(read_tsp_file(tsp_file_path), dtype = torch.float32, device = DEVICE)
    num_nodes = data.shape[0]
    distances = torch.cdist(data, data, p=2)
    _, nearest_indices = torch.topk(distances, K+1, dim=1, largest=False, sorted=True)
    _, farthest_indices = torch.topk(distances, num_nodes-(K+1), dim=1, largest=True, sorted=True)

    nearest_indices = nearest_indices[:,1:]
        

    params_dict = {'distance': distances,
                   'nearest_indices':nearest_indices,
                   'farthest_indices':farthest_indices,
                   'num_nodes':num_nodes}


    return params_dict

def read_tsp_file(file_path):
    cities = []
    with open(file_path, 'r') as f:
        content = f.readlines()
        start_parsing = False
        for line in content:
            if line.strip() == "NODE_COORD_SECTION":
                start_parsing = True
                continue
            if start_parsing:
                if line.strip() == "EOF":
                    break
                parts = line.strip().split()
                x = float(parts[1])
                y = float(parts[2])
                cities.append([x, y])
    return cities

