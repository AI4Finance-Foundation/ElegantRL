
import opt_einsum as oe
import numpy as np
import math

import numpy as np
import opt_einsum as oe
import math
import ast

input_str = None
tensors = None

N = 4
all_row_tensors_shape = []
count = 90
# 连接横线
for i in range(N):
    row_tensor_shape = []
    for j in range(N):
        if j==0:
            row_tensor_shape.append([count])
            count += 1
        elif j==N-1:
            row_tensor_shape.append([count-1])
        else:
            row_tensor_shape.append([count-1, count])
            count += 1
    all_row_tensors_shape.append(row_tensor_shape)
# 连接竖线
all_col_tensors_shape = []
for i in range(N):
    col_tensor_shape = []
    for j in range(N):
        if j==0:
            col_tensor_shape.append([count])
            count += 1
        elif j==N-1:
            col_tensor_shape.append([count-1])
        else:
            col_tensor_shape.append([count-1, count])
            count += 1
    all_col_tensors_shape.append(col_tensor_shape)
all_tensor_shape = []
output_str_shape = []
tensor_shape_for_construct = []
for i in range(N):
    for j in range(N):
        all_row_tensors_shape[i][j] = all_row_tensors_shape[i][j] + all_col_tensors_shape[j][i]
        all_row_tensors_shape[i][j].append(count)
        tensor_shape_for_construct.append([2]*len(all_row_tensors_shape[i][j]))
        output_str_shape.append(chr(count))
        count+=1

tensors = [np.random.rand(*shape) for shape in tensor_shape_for_construct]
output_str = ''.join(output_str_shape)

input_str_shape = []
for i in range(N):
    for j in range(N):
        tensor_str = ''
        for chn in all_row_tensors_shape[i][j]:
            tensor_str+=chr(chn)
        input_str_shape.append(tensor_str)
input_str = ','.join(input_str_shape)



# add string of all the possible indices
input_str += '->'
input_str += output_str
# print(input_str)
# print(input_str)

path_greedy = oe.contract_path(input_str, *tensors, optimize='greedy')[1]
print('greedy ', math.log2(path_greedy.opt_cost))

path_rand_greedy = oe.contract_path(input_str, *tensors, optimize='random-greedy')[1]
print('random greedy ', math.log2(path_rand_greedy.opt_cost))


optimizer = oe.DynamicProgramming(
    minimize='flops',    # optional: flops, size, write, combo, limit
    search_outer=True,  # search through outer products as well
    cost_cap=False,     # don't use cost-capping strategy
)

# path_dp = oe.contract_path(input_str, *tensors, optimize=optimizer)[1]
# print('DP ', math.log10(path_dp.opt_cost))

