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


def process_folder(result_folder_path, total_result_folder, include_time=False, comparison_method=None,
                   output_order=None, maxProblem=True):
    categories_with_ID = set()
    categories_without_ID = set()

    all_dirs = os.listdir(result_folder_path)
    for dir in all_dirs:
        dirs = result_folder_path + '/' + dir
        all_txts = os.listdir(dirs)
        for txt_ in all_txts:
            if 'ID' in txt_.upper():
                categories_with_ID.add(('_'.join(txt_.split('_')[:-2]) + '_'))
            else:
                categories_without_ID.add(txt_.split('_')[0])
    categories = categories_with_ID.union(categories_without_ID)
    for category in categories:
        summary_data = {}
        method_with_gap = []
        for dir in all_dirs:
            method_name = dir.split('_')[1:]
            if len(method_name) > 1:
                method_name = '_'.join(method_name)
            else:
                method_name = method_name[0]
            method_name = method_name

            dirs = result_folder_path + '/' + dir
            all_txts = os.listdir(dirs)
            for txt_ in all_txts:
                if txt_.startswith('.') or txt_.startswith('_'):
                    continue
                if txt_.endswith('.txt') and category in txt_ and category in categories_with_ID:
                    parts = txt_.split('_')
                    time_taken = float(parts[-1].split('.')[0]) if include_time else None

                    graph_id = parts[-2]
                    if graph_id not in summary_data:
                        summary_data[graph_id] = {}

                    file_path = os.path.join(dirs, txt_)
                    data = extract_data_from_file(file_path)

                    if graph_id in summary_data and f'{method_name}' in summary_data[graph_id]:
                        if not ((summary_data[graph_id][f'{method_name}'] > data['obj']) ^ maxProblem):
                            continue
                    if data['gap'] is not None:
                        method_with_gap.append(method_name)
                        summary_data[graph_id][f'{method_name}'] = data['obj']
                        summary_data[graph_id][f'{method_name}_Gap'] = data['gap']
                        summary_data[graph_id][f'{method_name}_Bound'] = data['obj_bound']
                        if include_time:
                            summary_data[graph_id][f'{method_name}_Time'] = time_taken
                    else:
                        summary_data[graph_id][method_name] = data['obj']
                        if include_time:
                            summary_data[graph_id][f'{method_name}_Time'] = time_taken

                elif txt_.endswith('.txt') and category in txt_ and category in categories_without_ID:
                    parts = txt_.split('_')
                    time_taken = float(parts[-1].split('.')[0]) if include_time else None
                    graph_id = parts[:-1]
                    if len(graph_id) > 1:
                        graph_id = '_'.join(graph_id)
                    else:
                        graph_id = graph_id[0]
                    if graph_id not in summary_data:
                        summary_data[graph_id] = {}

                    file_path = os.path.join(dirs, txt_)
                    data = extract_data_from_file(file_path)
                    if data['obj'] is not None:
                        if graph_id in summary_data and f'{method_name}' in summary_data[graph_id]:
                            if not ((summary_data[graph_id][f'{method_name}'] > data['obj']) ^ maxProblem):
                                continue
                        if data['gap'] is not None:
                            method_with_gap.append(method_name)
                            summary_data[graph_id][f'{method_name}'] = data['obj']
                            summary_data[graph_id][f'{method_name}_Gap'] = data['gap']
                            summary_data[graph_id][f'{method_name}_Bound'] = data['obj_bound']
                            if include_time:
                                summary_data[graph_id][f'{method_name}_Time'] = time_taken
                        else:
                            summary_data[graph_id][method_name] = data['obj']
                            if include_time:
                                summary_data[graph_id][f'{method_name}_Time'] = time_taken
            summary_data = dict(sorted(summary_data.items(), key=lambda x: int(''.join(filter(str.isdigit, x[0])))))

        output_folder = os.path.join(total_result_folder, f"{category.split('_')[0]}_results")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if category in categories_without_ID:
            df = pd.DataFrame.from_dict(summary_data, orient='index')

            if comparison_method and comparison_method in df.columns:
                # 插入目标函数比较值行
                comparison_rows = []
                for graph_id, row in df.iterrows():
                    if not graph_id.endswith("_Comparison") and not graph_id.endswith("_Time_Comparison"):
                        comparison_values = {}
                        # 对obj列进行比较
                        for col in df.columns:
                            # 不包含时间列和Gap、Bound列
                            if not col.endswith('_Time') and not any(t in col for t in ['Gap', 'Bound']):
                                if row[comparison_method] != 0 and not pd.isnull(row[col]):
                                    comparison_values[col] = (row[col] - row[comparison_method]) / row[
                                        comparison_method]
                                else:
                                    comparison_values[col] = None
                        comparison_df = pd.DataFrame(comparison_values,
                                                     index=[f"{graph_id}_Obj_Comparison_{comparison_method}"])
                        comparison_rows.append(comparison_df)

                for comparison_df in comparison_rows:
                    df = pd.concat([df, comparison_df])

                # 插入时间比较值行（如果有GUROBI_Time和其他_Time列）
                time_comparison_rows = []
                if f'{comparison_method}_Time' in df.columns:
                    for graph_id, row in df.iterrows():
                        if not graph_id.endswith("_Comparison") and not graph_id.endswith(
                                "_Time_Comparison") and not graph_id.endswith(f'{comparison_method}'):
                            time_comparison_values = {}
                            comparison_time = row[f'{comparison_method}_Time']
                            if comparison_time and comparison_time != 0:
                                for col in df.columns:
                                    if col.endswith('_Time'):
                                        if not pd.isnull(row[col]):
                                            time_comparison_values[col] = (row[col] - comparison_time) / comparison_time
                                        else:
                                            time_comparison_values[col] = None
                            # 只有在有任何时间列比较值时才添加这一行
                            if time_comparison_values:
                                time_comparison_df = pd.DataFrame(time_comparison_values, index=[
                                    f"{graph_id}_Time_Comparison_{comparison_method}"])
                                time_comparison_rows.append(time_comparison_df)

                    for time_comparison_df in time_comparison_rows:
                        df = pd.concat([df, time_comparison_df])

            if output_order:
                ordered_columns = []
                for method in output_order:
                    if method in df.columns:
                        if method in method_with_gap:
                            method_with_gap_columns = [col for col in [f"{method}", f"{method}_Gap", f"{method}_Bound",
                                                                       f"{method}_Time"] if col in df.columns]
                            ordered_columns.extend(method_with_gap_columns)
                            continue
                        ordered_columns.append(method)
                        if f"{method}_Time" in df.columns:
                            ordered_columns.append(f"{method}_Time")
                ordered_columns += [col for col in df.columns if col not in ordered_columns]
                df = df[ordered_columns]

            df.to_csv(os.path.join(output_folder, f'{category}_summary.csv'))
        else:
            df = pd.DataFrame.from_dict(summary_data, orient='index')

            df.loc['Average'] = df.mean()  # 添加平均值行

            if include_time:
                # 添加每种方法的平均时间行
                time_columns = [col for col in df.columns if col.endswith('_Time')]
                # for time_col in time_columns:
                #     df.loc['Average_Time', time_col] = df[time_col].mean()

                # 添加当前方法与对比方法的平均时间差
                if comparison_method and f'{comparison_method}_Time' in df.columns:
                    comparison_avg_time = df[f'{comparison_method}_Time'].mean()
                    for time_col in time_columns:
                        df.loc[f'Time_Comparison_{comparison_method}', time_col] = (df[
                                                                                        time_col].mean() - comparison_avg_time) / comparison_avg_time

            if comparison_method and comparison_method in df.columns:
                comparison_values = df.loc['Average']
                diff_row = {}
                for col in df.columns:
                    if not col.endswith('_Time') and not any(t in col for t in ['Gap', 'Bound']):
                        diff_row[col] = (comparison_values[col] - comparison_values[comparison_method]) / \
                                        comparison_values[comparison_method]
                df.loc[f'Obj_Comparison_{comparison_method}'] = pd.Series(diff_row)

            if output_order:
                ordered_columns = []
                for method in output_order:
                    if method in df.columns:
                        if method in method_with_gap:
                            method_with_gap_columns = [col for col in [f"{method}", f"{method}_Gap", f"{method}_Bound",
                                                                       f"{method}_Time"] if col in df.columns]
                            ordered_columns.extend(method_with_gap_columns)
                            continue
                        ordered_columns.append(method)
                        if f"{method}_Time" in df.columns:
                            ordered_columns.append(f"{method}_Time")
                ordered_columns += [col for col in df.columns if col not in ordered_columns]
                df = df[ordered_columns]

            df.to_csv(os.path.join(output_folder, f'{category}summary.csv'))


class Config():
    # result_folder_path中，文件夹的名称按照 datasetName_AlgName的方式写，只有一个下划线，否则报错。正确的写法，比如，synBA_GA，表示synBA数据集，GA算法。
    # result_folder_path = r'./result'# 替换为实际路径
    result_folder_path = r'D:\cs\RLSolver_data_result\result_maxcut'  # 替换为实际路径
    total_result_folder = r'D:\cs\RLSolver_data_result\result_maxcut统计excel'  # 替换为要存放结果的路径
    include_time = True  # 设置是否统计时间
    comparison_method = "gurobiQUBO"  # 设置对比的方法名称
    output_order = ["greedy", "SDP", "SA", "GA", "gurobiQUBO", "s2v", "iSCO", "MCPG"]  # 设置表格列的输出顺序
    maxProblem = True  # 若同一个数据集同一个方法有多个结果，是否保留最大值


if __name__ == "__main__":
    if os.path.exists(Config.total_result_folder):
        shutil.rmtree(Config.total_result_folder)  # 如果存在旧的结果文件夹，先删除
    os.makedirs(Config.total_result_folder)

    maxProblem = True
    process_folder(Config.result_folder_path, Config.total_result_folder,
                   include_time=Config.include_time, comparison_method=Config.comparison_method,
                   output_order=Config.output_order, maxProblem=Config.maxProblem)
