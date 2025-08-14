
# 顺序收缩在 GPU 上的时间: 0.014693023636937141
# 并行收缩在 GPU 上的时间: 0.0018282514065504074


import torch
import timeit
import string

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 生成包含复数的随机张量的列表
tensors = [torch.complex(torch.rand([2] * 27, device=device), torch.rand([2] * 27, device=device)) for _ in range(8)]

# 将输入数组转换为更易处理的形式
def transform_array(input_array):
    transformed_array = []
    temp_dict = {}
    counter = 0
    size_dict = {}
    tensors = []

    for tensor_id, connections in enumerate(input_array):
        transformed_array.append([])
        for connected_tensor in connections:
            edge_key = tuple(sorted((tensor_id, connected_tensor)))
            if edge_key not in temp_dict:
                temp_dict[edge_key] = string.ascii_lowercase[counter]
                size_dict[string.ascii_lowercase[counter]] = 2
                counter += 1
            transformed_array[tensor_id].append(temp_dict[edge_key])

        transformed_array[tensor_id] = ''.join(transformed_array[tensor_id])
        tensor_shape = [2 for _ in range(len(transformed_array[tensor_id]))]
        tensors.append(torch.complex(torch.rand(tensor_shape, device=device), torch.rand(tensor_shape, device=device)))

    return transformed_array, size_dict, tensors

# 示例输入数组
input_array = [[1, 2, 4], [0], [0], [2], [0, 5, 6], [4], [7], [6]]
transformed_tensors, size_dict, tensors = transform_array(input_array)

# 生成两个张量的 einsum 方程
def generate_einsum_equation(i, j):
    return transformed_tensors[i] + ',' + transformed_tensors[j] + '->' + transformed_tensors[i]

# 使用 torch.einsum 执行收缩操作的函数
def perform_contraction(tensors, contraction_order):
    for i, j in contraction_order:
        equation = generate_einsum_equation(i, j)
        tensors[i] = torch.einsum(equation, tensors[i], tensors[j])
    return tensors[0]

# 顺序收缩的收缩顺序
contraction_order_1 = [(0, 1), (2, 3), (0, 2), (4, 5), (6, 7), (4, 6), (0, 4)]

# 测量第一种情况的时间：顺序收缩
time_1 = timeit.timeit(lambda: perform_contraction(tensors.copy(), contraction_order_1), number=10)


# 并行收缩的收缩顺序
parallel_order = [(0, 1), (2, 3), (4, 5), (6, 7)]
second_order = [(0, 2), (4, 6)]
final_order = [(0, 4)]

# 执行第二种情况的函数：并行收缩
def parallel_contraction(tensors):
    for i, j in parallel_order:
        equation = generate_einsum_equation(i, j)
        tensors[i] = torch.einsum(equation, tensors[i], tensors[j])

    for i, j in second_order:
        equation = generate_einsum_equation(i, j)
        tensors[i] = torch.einsum(equation, tensors[i], tensors[j])

    equation = generate_einsum_equation(final_order[0][0], final_order[0][1])
    return torch.einsum(equation, tensors[final_order[0][0]], tensors[final_order[0][1]])

# 测量第二种情况的时间：并行收缩
time_2 = timeit.timeit(lambda: parallel_contraction(tensors.copy()), number=10)

# 打印结果
print("顺序收缩在 GPU 上的时间:", time_1)
print("并行收缩在 GPU 上的时间:", time_2)
