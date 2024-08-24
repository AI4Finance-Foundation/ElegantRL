import os
import re
import tqdm
# import numpy as np
import pandas as pd

# from evaluator import EncoderBase64

'''
还未整理，值得整理
读取 data_dir 的solution，然后输出一个df
'''


def time_limit_str(input_string):
    # 使用正则表达式提取浮点数
    matches = re.findall(r'[-+]?\d*\.\d+|\d+', input_string)
    return matches[0]


def extract_info_from_filename(file_name):
    parts = file_name.split('_')

    if parts[0] != 'powerlaw':
        parts[0] = f"{parts[0]}_{parts[1]}"
        del parts[1]
    info_dict = {
        'graph_type': parts[0],
        'num_nodes': int(parts[1]),
        'random_seed_id': int(parts[2][2:]),  # 从 'ID' 后的部分提取
        'exec_time': int(parts[3][:-4])  # 从文件扩展名前的部分提取
    }
    return info_dict


def read_key_value_file(directory, filename):
    file_path = f"{directory}/{filename}"
    result_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('//'):
                break
            key, value = map(str.strip, line[2:].split(':', 1))
            result_dict[key] = value

    result_dict['time_limit'] = time_limit_str(result_dict['time_limit'])
    result_dict = {key: float(value) for key, value in result_dict.items()}
    result_dict.update(extract_info_from_filename(filename))

    # enc = EncoderBase64(num_nodes=result_dict['num_nodes'])
    # x_bool = np.equal(np.array(list(map(int, solution_sequence))), 1)
    # result_dict['solution'] = enc.bool_to_str(x_bool=x_bool)
    # print(result_dict)
    return result_dict


def read_all_files_in_directory(directory):
    data_dict_list = []
    file_names = sorted(os.listdir(directory))
    for file_name in tqdm.tqdm(file_names):
        if file_name.endswith(".txt"):
            data_dict = read_key_value_file(directory, file_name)
            data_dict_list.append(data_dict)
    return data_dict_list


'''run'''


def collect_obj_value():
    # 例子
    graph_type = ['powerlaw', 'erdos_renyi', 'barabasi_albert'][2]
    data_dir = f'./data/syn_{graph_type}_result'
    save_csv = f'./data/syn_{graph_type}.csv'

    data_dicts = read_all_files_in_directory(data_dir)
    df = pd.DataFrame(data_dicts)

    desired_order = ['graph_type', 'num_nodes', 'random_seed_id', 'exec_time', 'obj', 'time_limit',
                     'running_duration', 'gap', 'obj_bound', ]  # 'solution']
    df = df[desired_order]  # 重新指定 DataFrame 的列顺序

    # 制定排序的列名优先级
    priority_columns = ['graph_type', 'num_nodes', 'random_seed_id', 'exec_time']
    # 使用 sort_values 方法进行排序
    df = df.sort_values(by=priority_columns)
    print(df[['graph_type', 'num_nodes', 'random_seed_id', 'obj', 'exec_time', ]])  # 'solution']])
    df.to_csv(save_csv, index=False)


def load_graph_info_from_data_dir(csv_path: str, csv_id: int) -> (str, str):
    # graph_type = ['ErdosRenyi', 'PowerLaw', 'BarabasiAlbert'][1]
    # csv_path = f'./data/syn_{graph_type}_3600.csv'

    # not elegant
    map_graph_name_to_type = {'erdos_renyi': 'ErdosRenyi', 'powerlaw': 'PowerLaw', 'barabasi_albert': 'BarabasiAlbert'}
    map_graph_type_to_name = {'ErdosRenyi': 'erdos_renyi', 'PowerLaw': 'powerlaw', 'BarabasiAlbert': 'barabasi_albert'}
    # not elegant

    df_row = pd.read_csv(csv_path).iloc[csv_id]
    df_row = df_row[['graph_type', 'num_nodes', 'random_seed_id', 'exec_time', 'obj']]

    num_nodes = df_row['num_nodes']
    random_seed_id = df_row['random_seed_id']
    graph_type = df_row['graph_type']
    if graph_type not in {'ErdosRenyi', 'PowerLaw', 'BarabasiAlbert'}:
        graph_type = map_graph_name_to_type[graph_type]
    assert graph_type in {'ErdosRenyi', 'PowerLaw', 'BarabasiAlbert'}

    txt_path = f"./data/syn_{graph_type}/{graph_type}_{num_nodes}_ID{random_seed_id}.txt"
    if not os.path.exists(txt_path):
        graph_name = map_graph_type_to_name[graph_type]
        txt_path = f"./data/syn_{graph_type}/{graph_name}_{num_nodes}_ID{random_seed_id}.txt"
    if not os.path.exists(txt_path):
        from graph_utils import load_graph_list
        graph_name = f"{graph_type}_{num_nodes}_ID{random_seed_id}"
        graph_list = load_graph_list(graph_name=graph_name)
        num_edges = len(graph_list)
        with open(txt_path, 'w') as file:
            file.write(f"{num_nodes} {num_edges}\n")
            for n0, n1, dt in graph_list:
                file.write(f"{n0 + 1} {n1 + 1} {dt}\n")  # 将node_id 由“从0开始”改为“从1开始”
    assert os.path.exists(txt_path)  # todo

    sim_name = f"{graph_type}_{num_nodes}_ID{random_seed_id}"
    return txt_path, sim_name


def collect_obj_value_of_3600sec():
    graph_type = ['powerlaw', 'erdos_renyi', 'barabasi_albert'][2]
    df = pd.read_csv(f'./data/syn_{graph_type}.csv')

    # 按 'graph_type', 'num_nodes', 'random_seed_id' 进行分组，并找到每组中 'exec_time' 最大的行
    result_df = df.loc[df.groupby(['graph_type', 'num_nodes', 'random_seed_id'])['exec_time'].idxmax()]
    result_df.to_csv(f'./data/syn_{graph_type}_3600.csv', index=False)


'''zzz'''


def load_graph_for_sim(df_row, device):  # todo 待处理完beta3的代码后，删除此函数
    if df_row is None:
        graph_type = ['powerlaw', 'erdos_renyi', 'barabasi_albert'][1]
        df = pd.read_csv(f'./data/syn_{graph_type}_3600.csv')
        df_row = df.iloc[0]

    graph_type = df_row['graph_type']
    num_nodes = df_row['num_nodes']
    random_seed_id = df_row['random_seed_id']
    txt_path = f"./data/syn_{graph_type}/{graph_type}_{num_nodes}_ID{random_seed_id}.txt"
    from maxcut_simulator import SimulatorMaxcut
    from graph_utils import load_graph_list_from_txt
    graph = load_graph_list_from_txt(txt_path=txt_path)
    sim = SimulatorMaxcut(sim_name=f"{graph_type}_{num_nodes}_ID{random_seed_id}", graph_list=graph, device=device)
    return sim


if __name__ == '__main__':
    collect_obj_value()
    collect_obj_value_of_3600sec()
