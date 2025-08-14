import numpy as np
import cotengra as ctg
import opt_einsum as oe
import math




dim_tensor = 2

# binary tree
height_tree  = 4
number_of_nodes = 2**height_tree - 1
number_of_leaves = 2**(height_tree-1)
root_shape = [dim_tensor, dim_tensor]
non_root_nodes_shape = [[dim_tensor, dim_tensor, dim_tensor] for _ in range(number_of_nodes - 1)]
tensor_shape =[root_shape] + non_root_nodes_shape
tensors = [np.random.rand(*shape) for shape in tensor_shape]

root_indices = [140, 141]
second_layer_indices = [[140,142,143], [141,144,145]]
tree_indices = [root_indices] + second_layer_indices
last_height = second_layer_indices
end_index = last_height[-1][-1]
print("root indices: ", root_indices)
print("height: ", 2)
print(second_layer_indices)
for h in range(3, height_tree+1):
    indices = []
    for node_i in range(2**(h-1)):
        paranet_node_i = (node_i)//2
        left_or_right = (node_i)%2 + 1
        parent_node_indices = last_height[paranet_node_i]
        child_indices = [parent_node_indices[left_or_right], end_index+1, end_index+2]
        end_index += 2
        indices.append(child_indices)
    last_height = indices.copy()
    tree_indices.extend(indices)
    print("height: ", h)
    print(indices)



output_indices = []
for node in last_height:
    output_indices.append(node[1])
    output_indices.append(node[2])
print("output indices:")
print(output_indices)
# print(output_indices)
# convert each shape to a string
tensor_indices_str = []
for node in tree_indices:
    tensor_indices_str.append(''.join([chr(i) for i in node]))

output_indices_str = ''.join([chr(i) for i in output_indices])
# print(output_indices_str)
inputs = tensor_indices_str
output = output_indices_str
size_dict={}
for i in range(140, end_index+1):
    size_dict[chr(i)] = 2




print(inputs)
print(output)
print(size_dict)

opt = ctg.HyperOptimizer(
    minimize='flops',
    reconf_opts={},
    progbar=True,
)
tree = opt.search(inputs, output, size_dict)
