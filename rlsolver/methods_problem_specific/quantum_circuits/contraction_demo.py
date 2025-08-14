import numpy as np
import torch
import timeit
import time

# 创建两个复数形式的24阶张量（每个indice为2）
tensor1_torch = torch.complex(torch.rand([2] * 30), torch.rand([2] * 30))
tensor2_torch = torch.complex(torch.rand([2] * 30), torch.rand([2] * 30))

# 移到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor1_torch = tensor1_torch.to(device)
tensor2_torch = tensor2_torch.to(device)

# Using torch.tensordot
tensordot_result_torch_gpu = timeit.timeit(lambda: torch.tensordot(tensor1_torch, tensor2_torch, dims=30), number=10)

# Using torch.matmul
matmul_result_torch = timeit.timeit(lambda: torch.matmul(tensor1_torch, tensor2_torch), number=10)


# 生成爱因斯坦积公式
def generate_einsum_equation(dimensions):
    assert dimensions > 0, "Dimensions should be positive"
    einsum_chars = [chr(97 + i) if i < 26 else chr(65 + i - 26) for i in range(dimensions)]
    equation_lhs = ''.join(einsum_chars)
    equation_rhs = equation_lhs
    return f"{equation_lhs},{equation_rhs}->"


# 为26阶以上的张量生成爱因斯坦积公式
einsum_equation = generate_einsum_equation(30)

# Using torch.einsum
einsum_result_torch_gpu = timeit.timeit(lambda: torch.einsum(einsum_equation, tensor1_torch, tensor2_torch), number=10)


# # 更新的 blockwise_parallel 函数
# def blockwise_parallel(tensor1_splits, tensor2_splits, split_dim=4):
#     results = []
#     for t1_split in tensor1_splits:
#         for t2_split in tensor2_splits:
#             result = torch.matmul(t1_split, t2_split)
#             results.append(result)
#     final_result = torch.cat(results, dim=split_dim)
#     return final_result


# 新的optimized_blockwise_parallel函数
def optimized_blockwise_parallel(tensor1, tensor2, num_chunks=8):
    chunk_size = tensor1.shape[0] // num_chunks
    results = []
    computation_times = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_chunks - 1 else tensor1.shape[0]
        chunk_a = tensor1[start_idx:end_idx]
        chunk_b = tensor2[start_idx:end_idx]

        # 执行张量收缩运算
        start_time_gpu = time.time()
        result_chunk = torch.tensordot(chunk_a, chunk_b, dims=30).to(device)
        end_time_gpu = time.time()

        results.append(result_chunk)
        computation_times.append(end_time_gpu - start_time_gpu)

    # 返回结果列表和计算时间列表
    return results, computation_times


# 分割张量并预先传输到GPU
def prepare_tensors(tensor, num_blocks=4, split_dim=4):
    tensor_splits = tensor.split(2, dim=split_dim)
    tensor_splits_gpu = [split.to(device) for split in tensor_splits]
    return tensor_splits_gpu


# 准备张量
tensor1_splits = prepare_tensors(tensor1_torch, num_blocks=4, split_dim=4)
tensor2_splits = prepare_tensors(tensor2_torch, num_blocks=4, split_dim=4)

# 测量优化后的方法时间
optimized_results, optimized_computation_times = optimized_blockwise_parallel(tensor1_torch, tensor2_torch,
                                                                              num_chunks=128)
max_computation_time = max(optimized_computation_times)

# Print results
print("PyTorch tensordot time:", tensordot_result_torch_gpu * 100)
print("PyTorch einsum time:", einsum_result_torch_gpu * 100)
print("PyTorch matmul time:", matmul_result_torch * 100)
print("Optimized GPU method time:", max_computation_time * 100)
