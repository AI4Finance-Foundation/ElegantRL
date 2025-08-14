import ast

# 从文件中读取contraction_ordering
# file_path = "N53M12.txt"
# file_path = "N53M14.txt"
# file_path = "N53M16.txt"
# file_path = "N53M18.txt"
file_path = "N53M20.txt"
with open(file_path, 'r') as file:
    content = file.read()
    contraction_ordering = ast.literal_eval(content)

# 查找并存储所有满足条件的组合
valid_combinations = []
i = 0
while i < len(contraction_ordering) - 2:
    order1 = contraction_ordering[i]
    order2 = contraction_ordering[i + 1]
    order3 = contraction_ordering[i + 2]

    a1, b1 = order1
    c1, d1 = order2
    e1, f1 = order3

    if ((a1 == e1 or b1 == e1) and (c1 == f1 or d1 == f1)):
        valid_combinations.append((order1, order2, order3))

    i += 1

# 合并组合
merged_combinations = []
i = 0
while i < len(valid_combinations):
    current = valid_combinations[i]

    # 检查是否可以与下一个组合合并
    if i + 1 < len(valid_combinations):
        next_combination = valid_combinations[i + 1]
        if not (set(current[0]).intersection(set(next_combination[0])) or
                set(current[1]).intersection(set(next_combination[1]))):
            # 合并组合并跳过下一个
            merged_combinations.append(
                (current[0], current[1], next_combination[0], next_combination[1], current[2], next_combination[2]))
            i += 1  # 跳过下一个组合
        else:
            merged_combinations.append(current)
    else:
        merged_combinations.append(current)

    i += 1

count = 0
for group in merged_combinations:
    print(f"符合条件的组 {count + 1}:")

    # 确定需要移动到第二行的对的数量
    num_to_move = len(group) // 3

    # 第一行
    first_line = group[:-num_to_move]
    print(" ".join(str(item) for item in first_line))

    # 第二行（如果有）
    if num_to_move > 0:
        second_line = group[-num_to_move:]
        print(" ".join(str(item) for item in second_line))

    print()
    count += 1

print(f"共找到{count}组满足条件的收缩对。")
