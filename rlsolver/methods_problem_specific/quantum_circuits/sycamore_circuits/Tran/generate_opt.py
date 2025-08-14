# Generate myhashmap&mycurrent of opt
import opt_einsum as oe
import numpy as np
import math

import numpy as np
import opt_einsum as oe
import math
import ast

def read_input_array(file_name):
    with open(file_name, 'r') as file:
        data = file.read()
        input_array = ast.literal_eval(data)
    return input_array

input_array = read_input_array('n53_m20.txt')

def transform_array(input_array):
    transformed_array = []
    temp_dict = {}
    counter = 97  # ASCII code for "a"
    size_dict = {}
    tensors = []
    for tensor_id, connections in enumerate(input_array):
        transformed_array.append([])
        for connected_tensor in connections:
            edge_key = tuple(sorted((tensor_id, connected_tensor)))
            if edge_key not in temp_dict:
                temp_dict[edge_key] = chr(counter)
                size_dict[chr(counter)] = 2
                counter += 1
            transformed_array[tensor_id].append(temp_dict[edge_key])
        transformed_array[tensor_id] = ''.join(transformed_array[tensor_id])
        tensor_shape = [2 for _ in range(len(transformed_array[tensor_id]))]
        tensors.append(np.random.rand(*tensor_shape))
    # print(temp_dict)

    return transformed_array, size_dict, tensors
#print(input_array)
input_str, size_dict, tensors = transform_array(input_array)
input_str = ','.join(input_str)
with open('myhashmap.txt', 'w', encoding='utf-8') as file:
    file.write(input_str)
# add string of all the possible indices
input_str += '->'
input_str += ''
# print(input_str)

path_rand_greedy = oe.contract_path(input_str, *tensors, optimize='random-greedy')[1]
print('random greedy ', math.log10(path_rand_greedy.opt_cost))
# print(path_rand_greedy)




current = []
for line in path_rand_greedy.contraction_list:
    inds, idx_rm, einsum_str, remaining, do_blas = line
    current.append(einsum_str)
with open('mycurrent.txt', 'w', encoding='utf-8') as file:
    for line in current:
        file.write(line + '\n')




path_rand_greedy = oe.contract_path(input_str, *tensors, optimize='random-greedy')[0]
print(path_rand_greedy)
