import numpy as np
import opt_einsum as oe
import math
import ast
import pickle as pkl


def read_input_array(file_name):
    with open(file_name, 'r') as file:
        data = file.read()
        input_array = ast.literal_eval(data)
    return input_array


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


def generate():
    input_array = read_input_array('./n53_m20.txt')

    # print(input_array)
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
    # print(path_rand_greedy.contraction_list)
    # print(path_rand_greedy)

    current = []
    for line in path_rand_greedy.contraction_list:
        inds, idx_rm, einsum_str, remaining, do_blas = line
        current.append(einsum_str)
    with open('mycurrent.txt', 'w', encoding='utf-8') as file:
        for line in current:
            file.write(line + '\n')


def process():
    with open('myhashmap.txt', 'r') as f:
        hashmap = f.readline()
        alphas = hashmap.split(',')

    orders = []
    lefts = []
    rights = []

    with open('mycurrent.txt', 'r') as f:  # read lines from current file
        lines = f.readlines()
        for line in lines:
            # each line has a format of "a , b -> c"
            # split the line into three parts
            ops, ans = line.replace('\n', '').split('->')
            a, b = ops.split(',')
            try:
                index_a = alphas.index(a)
            except:
                try:
                    ii = lefts.index(a)
                    print('a found in lefts')
                except:
                    try:
                        ii = rights.index(a)
                        print('a found in rights')
                    except:
                        print('a:##{}## not found'.format(a))
            try:
                index_b = alphas.index(b)
            except:
                try:
                    ii = lefts.index(b)
                    print('b found in lefts')
                except:
                    try:
                        ii = rights.index(b)
                        print('b found in rights')
                    except:
                        print('b:##{}## not found'.format(b))
            orders.append((index_a, index_b))
            alphas[index_a] = ans
            alphas[index_b] = ans
            lefts.append(a)
            lefts.append(b)
            rights.append(ans)

    with open('myorders.pkl', 'wb') as f:
        pkl.dump(orders, f)

    # load orders
    with open('myorders.pkl', 'rb') as f:
        orders = pkl.load(f)
        print(orders)


if __name__ == '__main__':
    generate()
    process()
