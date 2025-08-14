import ast

# 从文件中读取contraction_ordering
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
def can_merge(comb1, comb2):
    """检查两个组合是否有共同的元素"""
    flat_comb1 = set(sum([list(pair) for pair in comb1], []))
    flat_comb2 = set(sum([list(pair) for pair in comb2], []))
    return not flat_comb1.isdisjoint(flat_comb2)

def merge_combinations(combinations):
    """合并没有共同元素的相邻组合"""
    i = 0
    merged = []
    while i < len(combinations):
        current = combinations[i]
        if i + 1 < len(combinations) and not can_merge(current, combinations[i + 1]):
            # 合并当前组合与下一个组合
            next_combination = combinations[i + 1]
            merged_combination = current + next_combination
            merged.append(merged_combination)
            i += 2  # 跳过合并的组合
        else:
            merged.append(current)
            i += 1
    return merged

# 迭代优化直到不再有合并机会
while True:
    new_combinations = merge_combinations(valid_combinations)
    if len(new_combinations) == len(valid_combinations):
        break
    valid_combinations = new_combinations

# 打印最终结果
count = 0
for group in valid_combinations:
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
