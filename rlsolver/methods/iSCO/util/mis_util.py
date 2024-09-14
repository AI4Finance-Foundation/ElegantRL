import os

import pickle
import torch
from rlsolver.methods.iSCO.config.mis_config import *
def load_data(data_path):
    with open(data_path, 'rb') as f:
        g = pickle.load(f)
        num_nodes = g.number_of_nodes()
        num_edges = g.number_of_edges()
        edge_from = torch.zeros(num_edges, dtype=torch.long,device=DEVICE)
        edge_to = torch.zeros(num_edges, dtype=torch.long,device=DEVICE)
        for i, e in enumerate(g.edges(data=True)):
            x, y = e[0], e[1]
            edge_from[i] = x
            edge_to[i] = y
        params_dict = {'num_nodes': num_nodes,
                       'num_edges': num_edges,
                       'edge_from': edge_from,
                       'edge_to': edge_to}
    return params_dict

def write_result(data_directory,result,energy,running_duration,max_num_nodes):
    output_filename = '../../result/mis_iSCO'+'/result_' + os.path.basename(data_directory)
    output_filename = os.path.splitext(output_filename)[0]+'.txt'
    directory = os.path.dirname(output_filename)
    if not os.path.exists(directory):
        os.mkdir(directory)
    counter = 1
    while os.path.exists(output_filename):
        base, extension = os.path.splitext(output_filename)
        output_filename = f"{base}_{counter}{extension}"
        counter +=1
    with open(output_filename, 'w', encoding="UTF-8") as file:
        if energy is not None:
            file.write(f'// obj: {energy}\n')
            file. write(f'//running_duration:{running_duration}\n')
        for node in range(max_num_nodes):
            file.write(f'{node + 1} {int(result[node] + 1)}\n')