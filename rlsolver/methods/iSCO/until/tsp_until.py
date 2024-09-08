from rlsolver.methods.iSCO.config.tsp_config import *
import os
import torch
import numpy as np

def load_data(tsp_file_path,optimal_tour):

    # 通过相对路径获取 a.txt 的路径
    if tsp_file_path.endswith('.npy'):
        data = torch.tensor(np.load(tsp_file_path), dtype = torch.float32, device = DEVICE)[0]
        sols = torch.tensor(np.load(optimal_tour), dtype = torch.float32, device = DEVICE)[0]
        num_nodes = sols.shape[0]-1
    else:
        data = torch.tensor(read_tsp_file(tsp_file_path), dtype = torch.float32, device = DEVICE)
        sols = torch.tensor(read_opt_tour_file(optimal_tour), dtype = torch.long, device = DEVICE)
        num_nodes = data.shape[0]

    distances = torch.cdist(data, data, p=2)
    _, nearest_indices = torch.topk(distances, K+1, dim=1, largest=False, sorted=True)
    _, farthest_indices = torch.topk(distances, num_nodes-(K+1), dim=1, largest=True, sorted=True)

    nearest_indices = nearest_indices[:,1:]
        

    params_dict = {'distance': distances,
                   'nearest_indices':nearest_indices,
                   'sols':sols,
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

def read_opt_tour_file(file_path):
    tour = []
    with open(file_path, 'r') as f:
        content = f.readlines()
        start_parsing = False
        for line in content:
            if line.strip() == "TOUR_SECTION":
                start_parsing = True
                continue
            if start_parsing:
                if line.strip() == "-1" or line.strip() == "EOF":
                    break
                tour.append(int(line.strip()) - 1)  # 转换为从0开始的索引
    return tour



def write_result(data_directory,result, energy, running_duration,max_num_nodes):

    data_directory = os.path.dirname(DATA_PATH)

# 获取文件名，不带扩展名
    file_name_without_ext = os.path.splitext(os.path.basename(DATA_PATH))[0]
    output_filename = os.path.join('iSCO_result','tsp',"result_" + file_name_without_ext + '.txt')
    counter = 1
    while os.path.exists(output_filename):
        base, extension = os.path.splitext(output_filename)
        output_filename = f"{base}_{counter}{extension}"
        counter += 1
        
    with open(output_filename, 'w') as file:
        if energy is not None:
            file.write(f'// obj: {energy}\n')
            file.write(f'//running_duration: {running_duration}\n')
                              
        for node in range(max_num_nodes):
            file.write(f'{result[node] + 1}\n')