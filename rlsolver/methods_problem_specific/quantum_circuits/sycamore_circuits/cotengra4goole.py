import numpy as np
import cotengra as ctg
import opt_einsum as oe
import math




import ast

def read_input_array(file_name):
    with open(file_name, 'r') as file:
        data = file.read()
        input_array = ast.literal_eval(data)
    return input_array

input_array = read_input_array('sycamore/n53_m20.txt')

def transform_array(input_array):
    transformed_array = []
    temp_dict = {}
    counter = 97  # ASCII code for "a"
    size_dict = {}
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
    # print(temp_dict)

    return transformed_array, size_dict
#print(input_array)
input_str, size_dict = transform_array(input_array)
output_str = ''





# print(inputs)
# print(output)
# print(size_dict)

opt = ctg.HyperOptimizer(
    minimize='flops',
    reconf_opts={},
    progbar=True,
)
tree = opt.search(input_str, output_str, size_dict)
