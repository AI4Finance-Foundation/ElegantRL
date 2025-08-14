import torch
from numpy import random
import numpy as np

# 产生n个结果
for i in range(500):
    # 生成一个随机数矩阵
    matrix_S = random.randint(1, 21, size=(4, 4))
    matrix_mask1 = torch.zeros(4, 4)
    matrix_mask1[0, 0] = 1
    for i in range(1, 4):
        matrix_mask1[i, i - 1] = 1
        matrix_mask1[i, i] = 1
    matrix_S = np.multiply(matrix_mask1, matrix_S)
    matrix_2 = torch.ones_like(matrix_S)
    matrix_S += matrix_2
    matrix_S -= matrix_mask1

    # 21->32->43
    Order_1 = matrix_S[0][0] * matrix_S[1][0] * matrix_S[1][1] * matrix_S[2][1] \
              + matrix_S[0][0] * matrix_S[1][1] * matrix_S[2][1] * matrix_S[2][2] * matrix_S[3][2] \
              + matrix_S[0][0] * matrix_S[1][1] * matrix_S[2][2] * matrix_S[3][2] * matrix_S[3][3]
    # 21->43->32
    Order_2 = matrix_S[0][0] * matrix_S[1][0] * matrix_S[1][1] * matrix_S[2][1] \
              + matrix_S[2][1] * matrix_S[2][2] * matrix_S[3][2] * matrix_S[3][3] \
              + matrix_S[0][0] * matrix_S[1][1] * matrix_S[2][1] * matrix_S[2][2] * matrix_S[3][3]
    # 32->21->43
    Order_3 = matrix_S[1][0] * matrix_S[1][1] * matrix_S[2][1] * matrix_S[2][2] * matrix_S[3][2] \
              + matrix_S[0][0] * matrix_S[1][0] * matrix_S[1][1] * matrix_S[2][2] * matrix_S[3][2] \
              + matrix_S[0][0] * matrix_S[1][1] * matrix_S[2][2] * matrix_S[3][2] * matrix_S[3][3]
    # 32->43->21
    Order_4 = matrix_S[1][0] * matrix_S[1][1] * matrix_S[2][1] * matrix_S[2][2] * matrix_S[3][2] \
              + matrix_S[1][0] * matrix_S[1][1] * matrix_S[2][2] * matrix_S[3][2] * matrix_S[3][3] \
              + matrix_S[0][0] * matrix_S[1][0] * matrix_S[1][1] * matrix_S[2][2] * matrix_S[3][3]
    # 43->21->32    Order_5和Order_2结果相同
    Order_5 = matrix_S[2][1] * matrix_S[2][2] * matrix_S[3][2] * matrix_S[3][3] \
              + matrix_S[0][0] * matrix_S[1][0] * matrix_S[1][1] * matrix_S[2][1] \
              + matrix_S[0][0] * matrix_S[1][1] * matrix_S[2][1] * matrix_S[2][2] * matrix_S[3][3]
    # 43->32->21
    Order_6 = matrix_S[2][1] * matrix_S[2][2] * matrix_S[3][2] * matrix_S[3][3] \
              + matrix_S[1][0] * matrix_S[1][1] * matrix_S[2][1] * matrix_S[2][2] * matrix_S[3][3] \
              + matrix_S[0][0] * matrix_S[1][0] * matrix_S[1][1] * matrix_S[2][2] * matrix_S[3][3]
    order_list = [Order_1, Order_2, Order_3, Order_4, Order_5, Order_6]
    # Best_Reward = min(Order_1, Order_2, Order_3, Order_4, Order_5, Order_6)
    min_index = np.argmin(order_list)
    # Its_Order = argmin(list) a =
    print(matrix_S.numpy())
    # print("Order_1 is", Order_1)
    # print("Order_2 is", Order_2)
    # print("Order_3 is", Order_3)
    # print("Order_4 is", Order_4)
    # print("Order_5 is", Order_5)
    # print("Order_6 is", Order_6)
    # 如果输出Order有Order_2   ，则代表 也输出Order_5
    print("The best Reward is", order_list[min_index].numpy(), f"and it's contraction order is order_{min_index + 1}")