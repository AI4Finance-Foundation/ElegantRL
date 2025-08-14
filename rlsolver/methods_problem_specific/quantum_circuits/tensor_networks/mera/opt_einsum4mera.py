import opt_einsum as oe
import numpy as np
import math


dim_tensor = 2

# binary tree
height_tree  = 5
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
# print("root indices: ", root_indices)
# print("height: ", 2)
# print(second_layer_indices)
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
    # print("height: ", h)
    # print(indices)



# transfer to mera network
mid_tensor_list = []
last_mid_height = None
add_tensors_num = 0
for h in range(2, height_tree+1):
    indices = []
    num_nodes = 2**(h-1) - 1
    for node_i in range(num_nodes):
        # print(2**(h-2) + node_i)
        left_parent_node_i = tree_indices[2**(h-2) + node_i]
        right_parent_node_i = tree_indices[2**(h-2) + node_i + 1]
        left_parent_node_i[2] = end_index + 1
        right_parent_node_i[1] = end_index + 2
        if h!=height_tree:
            left_child_node_i = tree_indices[2*(2**(h-2) + node_i + 1)]
            right_child_node_i = tree_indices[2*(2**(h-2) + node_i + 1)+1]
            left_child_node_i[0] = end_index + 3
            right_child_node_i[0] = end_index + 4
        add_index = [end_index + i + 1 for i in range(4)]
        mid_tensor_list.append(add_index)
        indices.append(add_index)
        end_index += 4
        add_tensors_num += 1
    last_mid_height = indices.copy()
tree_indices.extend(mid_tensor_list)
# print('input')
# print(tree_indices)
output_indices = []
output_indices.append(last_height[0][1])
output_indices.append(last_height[-1][2])
for node in last_mid_height:
    output_indices.append(node[2])
    output_indices.append(node[3])
# print("output indices:")
# print(output_indices)
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

tensors.extend([np.random.rand(2,2,2,2) for _ in range(add_tensors_num)])

path_greedy = oe.contract_path(einsum_str, *tensors, optimize='greedy')[1]
print('greedy ', math.log2(path_greedy.opt_cost))

path_rand_greedy = oe.contract_path(einsum_str, *tensors, optimize='random-greedy')[1]
print('random greedy ', math.log2(path_rand_greedy.opt_cost))






