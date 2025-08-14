import numpy as np
import cotengra as ctg
import opt_einsum as oe
import math

# inputs, output, shapes, size_dict = ctg.utils.lattice_equation([10, 10])
# arrays = [np.random.uniform(size=shape) for shape in shapes]
input_str = None
tensors = None

n = 10
n_dim_tensor = 3
dim_tensor = 2
tensor_shape = [[dim_tensor, dim_tensor, dim_tensor] for _ in range(n)]
tensor_shape[0] = [dim_tensor, dim_tensor]
tensor_shape[-1] = [dim_tensor, dim_tensor]
tensors = [np.random.rand(*shape) for shape in tensor_shape]
print('shape of each tensor')
for i in range(n):
    print(tensor_shape[i], ' ', end='')
print()

# Build the enum string
size_dict = {}
possible_indices = [chr(i) for i in range(ord('a'), ord('z') + 1)]
possible_indices.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
if n > 52:
    possible_indices.extend([chr(i + 140) for i in range(0, (n - 52) * 3)])

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

output = ''
str += ',{}{}'.format(possible_indices[last_index], possible_indices[last_index + 1])
inputs.append('{}{}'.format(possible_indices[last_index], possible_indices[last_index + 1]))
size_dict[possible_indices[last_index]] = 2
size_dict[possible_indices[last_index + 1]] = 2
contracted_dim_index.append(last_index + 1)
# add string of all the possible indices
str += '->'
for i in range(len(contracted_dim_index)):
    str += possible_indices[contracted_dim_index[i]]
    output += possible_indices[contracted_dim_index[i]]
print(str)

print(inputs)
print(output)
print(size_dict)

opt = ctg.HyperOptimizer(
    minimize='flops',
    reconf_opts={},
    progbar=True,
)
tree = opt.search(inputs, output, size_dict)
