
import opt_einsum as oe
import numpy as np
import math
# 最大支持1021
n=10
n_dim_tensor = 3
dim_tensor = 2
tensor_shape = [[dim_tensor, dim_tensor, dim_tensor] for _ in range(n)]
tensor_shape[0] = [dim_tensor, dim_tensor]
tensor_shape[-1] = [dim_tensor, dim_tensor]
tensors = [np.random.rand(*shape) for shape in tensor_shape]

# Build the enum string
size_dict = {}
possible_indices = [chr(i) for i in range(ord('a'), ord('z')+1)]
possible_indices.extend([chr(i) for i in range(ord('A'), ord('Z')+1)])
if n > 26:
    possible_indices.extend([chr(i + 140) for i in range(0, (n - 26) * 3)])


str = '{}{}'.format(possible_indices[0], possible_indices[1])
inputs = []
last_index = 1
inputs.append('{}{}'.format(possible_indices[0], possible_indices[1]))
size_dict[possible_indices[0]] = 2
size_dict[possible_indices[1]] = 2
contracted_dim_index = [0]
for i in range(1, n - 1):
    size_dict[possible_indices[last_index]] = 2
    size_dict[possible_indices[last_index + 1]] = 2
    size_dict[possible_indices[last_index + 2]] = 2
    str += ',{}{}{}'.format(possible_indices[last_index], possible_indices[last_index + 1],
                            possible_indices[last_index + 2])
    inputs.append('{}{}{}'.format(possible_indices[last_index], possible_indices[last_index + 1],
                                  possible_indices[last_index + 2]))
    contracted_dim_index.append(last_index + 1)
    last_index += 2
    

str += ',{}{}'.format(possible_indices[last_index], possible_indices[last_index+1])
contracted_dim_index.append(last_index+1)
# add string of all the possible indices
str += '->'
for i in range(len(contracted_dim_index)):
    str += possible_indices[contracted_dim_index[i]]
print(str)

path_greedy = oe.contract_path(str, *tensors, optimize='greedy')[1]
print('greedy ', math.log10(path_greedy.opt_cost))

path_rand_greedy = oe.contract_path(str, *tensors, optimize='random-greedy')[1]
print('random greedy ', math.log10(path_rand_greedy.opt_cost))


optimizer = oe.DynamicProgramming(
    minimize='flops',    # optional: flops, size, write, combo, limit
    search_outer=True,  # search through outer products as well
    cost_cap=False,     # don't use cost-capping strategy
)

# path_dp = oe.contract_path(str, *tensors, optimize=optimizer)[1]
# print('DP ', math.log10(path_dp.opt_cost))

