import numpy as np
import cotengra as ctg
import opt_einsum as oe
import math




input_str = None
tensors = None

N = 4
all_row_tensors_shape = []
count = 90
size_dict = {}
# 连接横线
for i in range(N):
    row_tensor_shape = []
    for j in range(N):
        if j==0:
            row_tensor_shape.append([count])
            size_dict[chr(count)] = 2
            count += 1
        elif j==N-1:
            row_tensor_shape.append([count-1])
        else:
            row_tensor_shape.append([count-1, count])
            size_dict[chr(count)] = 2
            count += 1
    all_row_tensors_shape.append(row_tensor_shape)
# 连接竖线
all_col_tensors_shape = []
for i in range(N):
    col_tensor_shape = []
    for j in range(N):
        if j==0:
            col_tensor_shape.append([count])
            size_dict[chr(count)] = 2
            count += 1
        elif j==N-1:
            col_tensor_shape.append([count-1])
        else:
            col_tensor_shape.append([count-1, count])
            size_dict[chr(count)] = 2
            count += 1
    all_col_tensors_shape.append(col_tensor_shape)
all_tensor_shape = []
output_str_shape = []
tensor_shape_for_construct = []
for i in range(N):
    for j in range(N):
        all_row_tensors_shape[i][j] = all_row_tensors_shape[i][j] + all_col_tensors_shape[j][i]
        all_row_tensors_shape[i][j].append(count)
        size_dict[chr(count)] = 2
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
input_str = input_str_shape







# print(inputs)
# print(output)
# print(size_dict)

opt = ctg.HyperOptimizer(
    minimize='flops',
    reconf_opts={},
    progbar=True,
)
tree = opt.search(input_str, output_str, size_dict)
