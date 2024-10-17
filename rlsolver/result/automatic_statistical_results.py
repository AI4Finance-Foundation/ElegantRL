import copy
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
                if file_name.startswith('.'):
                    continue
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
                        node_count = int(parts[1])
                        id_number = int(parts[2][2:])

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

        output_folder = os.path.join(total_result_folder, f'{category}_results')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if category == 'gset':
            df = pd.DataFrame.from_dict(summary_data, orient='index')
            df.index.name = 'Graph'
            df = df.sort_index()
            bound_values = df['Bound']
            df.drop(columns=['Bound'], inplace=True)

            # 计算每一行的最大值，忽略 'Bound' 列
            df['BEST'] = df.max(axis=1)

            # 将 'Bound' 列添加回 DataFrame
            df['Bound'] = bound_values
            # 定义排序顺序
            sort_order = ['GA', 'GREEDY', 'SA', 'SDP', 'GUROBI', 'Gap', 'Bound', 'BEST']
            existing_columns = [col for col in sort_order if col in df.columns]  # 按排序顺序对DataFrame进行排序
            df = df[existing_columns]
            df.to_excel(os.path.join(output_folder, 'gset_summary.xlsx'))
        else:
            for node_count, id_data in summary_data.items():
                df = pd.DataFrame.from_dict(id_data, orient='index')
                df.index.name = 'ID'
                df.sort_index()
                bound_values = df['Bound']
                df.drop(columns=['Bound'], inplace=True)

                # 计算每一行的最大值，忽略 'Bound' 列
                df['BEST'] = df.max(axis=1)

                # 将 'Bound' 列添加回 DataFrame
                df['Bound'] = bound_values
                # 定义排序顺序
                sort_order = ['GA', 'GREEDY', 'SA', 'SDP', 'GUROBI', 'Gap', 'Bound', 'BEST']
                existing_columns = [col for col in sort_order if col in df.columns]
                # 按排序顺序对DataFrame进行排序
                df = df[existing_columns]
                df.to_excel(os.path.join(output_folder, f'{category}_Nodes_{node_count}_summary.xlsx'))

def rename_files_with_prefix(old_prefix: str, new_prefix: str, directory: str):
    file_names = os.listdir(directory)
    for file_name in file_names:
        if file_name.startswith(old_prefix):
            new_file_name = copy.deepcopy(file_name)
            new_file_name = new_file_name.replace(old_prefix, new_prefix)
            file = directory + '/' + file_name
            new_file = directory + '/' + new_file_name
            os.rename(file, new_file)

if __name__ == "__main__":

    exe_rename = True
    if exe_rename:
        # old_prefix = 'barabasi_albert'
        # new_prefix = 'BA'
        old_prefix = 'erdos_renyi'
        new_prefix = 'ER'
        # old_prefix = 'powerlaw'
        # new_prefix = 'PL'
        directory = r'D:/cs/RLSolver_data_result/result_maxcut/syn_ER_gurobi_QUBO'
        rename_files_with_prefix(old_prefix, new_prefix, directory)

    exe_stat = True
    if exe_stat:
        result_folder_path = r'D:/cs/RLSolver_data_result/result_maxcut'  # 替换为实际路径
        total_result_folder = r'./output'  # 替换为要存放结果的路径

        if os.path.exists(total_result_folder):
            shutil.rmtree(total_result_folder)  # 如果存在旧的结果文件夹，先删除
        os.makedirs(total_result_folder)

        process_folder(result_folder_path, total_result_folder)
