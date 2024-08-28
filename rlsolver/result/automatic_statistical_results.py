import os
import shutil
import pandas as pd


def extract_data_from_file(file_path):
    data = {
        "obj": None,
        "gap": None,
        "obj_bound": None
    }

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("// obj:"):
                data["obj"] = float(line.split()[-1])
            elif line.startswith("// gap:"):
                data["gap"] = float(line.split()[-1])
            elif line.startswith("// obj_bound:"):
                data["obj_bound"] = float(line.split()[-1])
    return data


def process_folder(result_folder_path, total_result_folder):
    all_dirs = os.listdir(result_folder_path)

    # 分类处理不同类型的文件夹
    categories = {
        'gset': [d for d in all_dirs if d.startswith('gset')],
        'BA': [d for d in all_dirs if 'BA' in d.upper()],
        'ER': [d for d in all_dirs if 'ER' in d.upper()],
        'PL': [d for d in all_dirs if 'PL' in d.upper()]
    }

    for category, dirs in categories.items():
        summary_data = {}

        for dir_name in dirs:
            dir_path = os.path.join(result_folder_path, dir_name)
            method_name = dir_name.split('_')[-1].upper()

            for file_name in os.listdir(dir_path):
                if file_name.endswith('.txt'):
                    parts = file_name.split('_')

                    if category == 'gset':
                        graph_id = parts[1]
                        if graph_id not in summary_data:
                            summary_data[graph_id] = {}
                        file_path = os.path.join(dir_path, file_name)
                        data = extract_data_from_file(file_path)
                        if 'QUBO' in method_name:
                            summary_data[graph_id][f'GUROBI'] = data['obj']
                            summary_data[graph_id][f'Gap'] = data['gap']
                            summary_data[graph_id][f'Bound'] = data['obj_bound']
                        else:
                            summary_data[graph_id][method_name] = data['obj']

                    else:
                        node_count = int(parts[2]) if category != 'PL' else int(parts[1])
                        id_number = int(parts[3][2:]) if category != 'PL' else int(parts[2][2:])

                        if node_count not in summary_data:
                            summary_data[node_count] = {f'ID_{i}': {} for i in range(30)}

                        if f'ID_{id_number}' not in summary_data[node_count]:
                            summary_data[node_count][f'ID_{id_number}'] = {}

                        file_path = os.path.join(dir_path, file_name)
                        data = extract_data_from_file(file_path)

                        if 'QUBO' in method_name:
                            summary_data[node_count][f'ID_{id_number}'][f'GUROBI'] = data['obj']
                            summary_data[node_count][f'ID_{id_number}'][f'Gap'] = data['gap']
                            summary_data[node_count][f'ID_{id_number}'][f'Bound'] = data['obj_bound']
                        else:
                            summary_data[node_count][f'ID_{id_number}'][method_name] = data['obj']

        # 为每个类型生成对应的结果文件夹和CSV文件
        output_folder = os.path.join(total_result_folder, f'{category}_results')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if category == 'gset':
            df = pd.DataFrame.from_dict(summary_data, orient='index')
            df.index.name = 'Graph'
            df = df.sort_index()
            df.to_csv(os.path.join(output_folder, 'gset_summary.csv'))
        else:
            for node_count, id_data in summary_data.items():
                df = pd.DataFrame.from_dict(id_data, orient='index')
                df.index.name = 'ID'
                df.to_csv(os.path.join(output_folder, f'{category}_Nodes_{node_count}_summary.csv'))


if __name__ == "__main__":
    result_folder_path = r'./result'  # 替换为实际路径
    total_result_folder = r'./output'  # 替换为要存放结果的路径

    if os.path.exists(total_result_folder):
        shutil.rmtree(total_result_folder)  # 如果存在旧的结果文件夹，先删除
    os.makedirs(total_result_folder)

    process_folder(result_folder_path, total_result_folder)
