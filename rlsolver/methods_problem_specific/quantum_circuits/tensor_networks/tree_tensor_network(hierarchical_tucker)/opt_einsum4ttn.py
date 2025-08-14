import opt_einsum as oe
import numpy as np
import math


dim_tensor = 2

# binary tree
height_tree  = 8
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

# use , to join the tensor_indices_str
einsum_str = ','.join(tensor_indices_str)
# use -> to join the output_indices_str
einsum_str += '->' + output_indices_str


path_greedy = oe.contract_path(einsum_str, *tensors, optimize='greedy')[1]
print('greedy ', math.log10(path_greedy.opt_cost))

path_rand_greedy = oe.contract_path(einsum_str, *tensors, optimize='random-greedy')[1]
print('random greedy ', math.log10(path_rand_greedy.opt_cost))





