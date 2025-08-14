import json
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_scatter(logger_save_path):
    obj_values = []
    time_values = []
    time_step_values = []

    with open(logger_save_path, 'r') as f:
        for line in f:
            # 忽略注释行（以'//'开头的行）
            if line.startswith("//"):
                continue

            # 拆分每行数据并将其转换为浮动数
            obj, time_, time_step = map(float, line.split())

            # 将值添加到对应的列表
            obj_values.append(obj)
            time_values.append(time_)
            time_step_values.append(time_step)

        # 使用matplotlib绘图
        plt.figure(figsize=(16, 6))

        # 绘制obj随时间变化的图
        plt.subplot(2, 1, 1)
        plt.plot(time_values, obj_values, marker='o', color='b')
        plt.xlabel('Time')
        plt.ylabel('Obj')
        plt.title('Obj vs Time')
        plt.xlim(0, max(time_values))

        # 绘制obj随time_step变化的图
        plt.subplot(2, 1, 2)
        plt.plot(time_step_values, obj_values, marker='o', color='r')
        plt.xlabel('Time Step')
        plt.ylabel('Obj')
        plt.title('Obj vs Time Step')
        plt.xlim(0, max(time_step_values))

        plt.tight_layout()

        plot_save_path = os.path.splitext(logger_save_path)[0] + '.png'
        plot_save_path = plot_save_path.replace('result/eeco', 'result/eeco/plot')
        plot_dir = os.path.dirname(plot_save_path)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(plot_save_path, dpi=300)


def read_data_from_file(file_path):
    """从文件中读取数据并返回采样速度和时间"""
    times = []
    sampling_speeds = []

    with open(file_path, 'r') as f:
        for line in f:
            # 跳过注释行
            if line.startswith("//"):
                continue
            # 解析数据：采样速度，时间，时间步
            parts = line.split()
            if len(parts) >= 3:
                sampling_speeds.append(float(parts[0]))  # 采样速度
                times.append(float(parts[2]))  # 时间

    return times, sampling_speeds


def smooth_data(data, window_size):
    """ 平滑数据，使用滑动窗口平均 """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_sampling_speed(folder_path, max_time, window_size=5, xticks_stride=5, smooth=False):
    sampling_speeds = {}

    # 读取文件夹中的每个 JSON 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):  # 只处理 .json 文件
            file_path = os.path.join(folder_path, filename)

            # 打开并读取 JSON 文件
            with open(file_path, 'r') as f:
                data = json.load(f)

            # 提取 n_sims 和 sampling_speed
            n_sims = data.get('n_sims')
            sampling_speed = data.get('sampling_speed')

            if n_sims is not None and sampling_speed is not None:
                # 将采样速度数据按时间升序排列
                times = sorted(sampling_speed.keys(), key=float)  # 时间按升序排列
                speeds = [sampling_speed[t] for t in times]

                # 将每个 n_sims 对应的时间和采样速度存入字典
                if n_sims not in sampling_speeds:
                    sampling_speeds[n_sims] = {'times': [], 'speeds': []}

                sampling_speeds[n_sims]['times'].append(times)
                sampling_speeds[n_sims]['speeds'].append(speeds)

    # 按 n_sims 进行排序
    sorted_sims = sorted(sampling_speeds.keys())  # 获取排序后的 n_sims

    # 绘制图表
    plt.figure(figsize=(13, 6))
    for n_sims in sorted_sims:  # 按照排序后的 n_sims 进行遍历
        data = sampling_speeds[n_sims]

        # 获取时间和采样速度
        times = list(map(float, data['times'][0]))  # 将时间从字符串转换为浮动数值
        avg_speeds = [sum(speeds) / len(speeds) for speeds in zip(*data['speeds'])]  # 计算每个时间点的平均采样速度

        # 如果需要平滑，应用平滑窗口
        if smooth:
            smoothed_speeds = smooth_data(avg_speeds, window_size)
            label = f'cpu-env{n_sims}(ECO)' if n_sims == 1 else f'gpu-env{n_sims}(Ours)'
            plt.plot(times[:len(smoothed_speeds)], smoothed_speeds, label=label)
        else:
            label = f'cpu-env{n_sims}(ECO)' if n_sims == 1 else f'gpu-env{n_sims}(Ours)'
            plt.plot(times, avg_speeds, label=label)

    # 设置图表的标签和标题
    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel('Sampling Speed (samples/second)', fontsize=14)
    # plt.title('Sampling Speed vs Time for Different n_sims')

    # 设置 x 轴刻度
    plt.xticks(np.arange(0, max_time, xticks_stride // 2))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlim(0, max_time)

    # 显示图例（图例顺序按照 n_sims 排序）
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

    # 调整布局以适应图例
    plt.tight_layout()

    # 保存图表
    plt.savefig(folder_path + "/sampling_speed.png", dpi=300)


def plot_obj_vs_time(folder_path, smooth=False, window_size=5, max_time=100):
    """绘制不同环境数量下的obj随时间变化的图"""
    # 创建一个图表
    envs_obj_vs_time = {}

    # 遍历文件夹中的 JSON 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            # 读取每个 JSON 文件
            with open(os.path.join(folder_path, filename), 'r') as f:
                data = json.load(f)

                # 提取环境数量 (n_sims)
                n_sims = data.get("n_sims")

                # 提取 obj_vs_time 字典并将时间和 obj 对应起来
                obj_vs_time = data.get("obj_vs_time", {})

            if n_sims is not None and obj_vs_time is not None:

                # 对 obj_vs_time 按时间排序并提取 obj 值
                times = sorted(obj_vs_time.keys(), key=float)
                obj_values = [obj_vs_time[time][0] for time in times]

                if n_sims not in envs_obj_vs_time:
                    envs_obj_vs_time[n_sims] = {'times': [], 'obj': []}
                    # 保存数据到字典中
                envs_obj_vs_time[n_sims]['times'].append(times)
                envs_obj_vs_time[n_sims]['obj'].append(obj_values)
    # 创建图形
    plt.figure(figsize=(25, 6))

    # 为每个环境数量绘制 obj 随时间变化的曲线
    for n_sims, data in envs_obj_vs_time.items():
        times = list(map(float, data['times'][0]))
        objs = list(map(float, data['obj'][0]))
        if smooth:
            smoothed_objs = smooth_data(objs, window_size)
            if n_sims == 1:
                plt.plot(times[:len(smoothed_objs)], smoothed_objs, label=f'cpu-env{n_sims}(ECO)')
            else:
                plt.plot(times[:len(smoothed_objs)], smoothed_objs, label=f'gpu-env{n_sims}(Ours)')
        else:
            if n_sims == 1:
                plt.plot(times, objs, label=f'cpu-env{n_sims}(ECO)')
            else:
                plt.plot(times, objs, label=f'gpu-env{n_sims}(Ours)')

    # 添加图例
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
    plt.xlim(0, max_time)

    # 添加标题和标签
    plt.title('Object vs Time for Different n_sims')
    plt.xlabel('Time(second)', fontsize=14)
    plt.ylabel('Objective Value', fontsize=14)
    plt.savefig(folder_path + "/obj_vs_time.png", dpi=300)


def plot_loss(folder_path, smooth=False, window_size=5, max_time=100):
    """绘制不同环境数量下的obj随时间变化的图"""
    # 创建一个图表
    envs_loss_vs_time = {}

    # 遍历文件夹中的 JSON 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            # 读取每个 JSON 文件
            with open(os.path.join(folder_path, filename), 'r') as f:
                data = json.load(f)

                n_sims = data.get("n_sims")

                loss_vs_time = data.get("Loss", {})

            if n_sims is not None and loss_vs_time is not None:

                times = sorted(loss_vs_time.keys(), key=float)
                losses = [loss_vs_time[time][0] for time in times]

                if n_sims not in envs_loss_vs_time:
                    envs_loss_vs_time[n_sims] = {'times': [], 'losses': []}
                envs_loss_vs_time[n_sims]['times'].append(times)
                envs_loss_vs_time[n_sims]['losses'].append(losses)
    # 创建图形
    plt.figure(figsize=(25, 6))

    # 为每个环境数量绘制 obj 随时间变化的曲线
    for n_sims, data in envs_loss_vs_time.items():
        times = list(map(float, data['times'][0]))
        objs = list(map(float, data['losses'][0]))
        if smooth:
            smoothed_objs = smooth_data(objs, window_size)
            if n_sims == 1:
                plt.plot(times[:len(smoothed_objs)], smoothed_objs, label=f'cpu-env{n_sims}(ECO)')
            else:
                plt.plot(times[:len(smoothed_objs)], smoothed_objs, label=f'gpu-env{n_sims}(Ours)')
        else:
            if n_sims == 1:
                plt.plot(times, objs, label=f'cpu-env{n_sims}(ECO)')
            else:
                plt.plot(times, objs, label=f'gpu-env{n_sims}(Ours)')

    # 添加图例
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
    plt.xlim(0, max_time)

    # 添加标题和标签
    plt.title('Loss vs Time for Different n_sims')
    plt.xlabel('Time(second)', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.savefig(folder_path + "/loss_vs_time.png", dpi=300)


def plot_inference_obj_vs_time(folder_path, smooth=False, window_size=5, max_time=100):
    """绘制不同环境数量下的obj随时间变化的图"""
    # 创建一个图表
    envs_obj_vs_time = {}

    # 遍历文件夹中的 JSON 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            # 读取每个 JSON 文件
            with open(os.path.join(folder_path, filename), 'r') as f:
                data = json.load(f)

                # 提取环境数量 (n_sims)
                n_sims = data.get("n_sims")

                # 提取 obj_vs_time 字典并将时间和 obj 对应起来
                obj_vs_time = data.get("obj_vs_time", {})

            if n_sims is not None and obj_vs_time is not None:

                # 对 obj_vs_time 按时间排序并提取 obj 值
                for graph_name, obj_vs_time in obj_vs_time.items():
                    times = sorted(obj_vs_time.keys(), key=float)
                    obj_values = [obj_vs_time[time] for time in times]
                    if graph_name not in envs_obj_vs_time:
                        envs_obj_vs_time[graph_name] = {}
                    if n_sims not in envs_obj_vs_time[graph_name]:
                        envs_obj_vs_time[graph_name][n_sims] = {'times': [], 'obj': []}
                        # 保存数据到字典中
                    envs_obj_vs_time[graph_name][n_sims]['times'].append(times)
                    envs_obj_vs_time[graph_name][n_sims]['obj'].append(obj_values)

    for graph, graph_obj_vs_time in envs_obj_vs_time.items():
        # 创建图形
        plt.figure(figsize=(13, 6))

        # 为每个环境数量绘制 obj 随时间变化的曲线
        for n_sims, data in graph_obj_vs_time.items():
            times = list(map(float, data['times'][0]))
            objs = list(map(float, data['obj'][0]))
            if smooth:
                smoothed_objs = smooth_data(objs, window_size)
                if n_sims == 1:
                    plt.plot(times[:len(smoothed_objs)], smoothed_objs, label=f'cpu-env{n_sims}(ECO)')
                else:
                    plt.plot(times[:len(smoothed_objs)], smoothed_objs, label=f'gpu-env{n_sims}(Ours)')
            else:
                if n_sims == 1:
                    plt.plot(times, objs, label=f'cpu-env{n_sims}(ECO)')
                else:
                    plt.plot(times, objs, label=f'gpu-env{n_sims}(Ours)')

        # 添加图例
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
        plt.xlim(0, max_time)

        # 添加标题和标签
        plt.title(f'{graph.split(".")[0]} Object vs Time for Different n_sims')
        plt.xlabel('Time(second)', fontsize=14)
        plt.ylabel('Object Value', fontsize=14)
        plt.tight_layout()
        plt.savefig(folder_path + "/obj_vs_time.png", dpi=300)


def run():
    inference_obj_vs_time_folder = '/home/shixi/project/eeco_2_27/RLSolver-master/rlsolver/result/inference_obj_vs_time'
    loss_folder = "RLSolver-master/rlsolver/result/loss_vs_time"
    obj_vs_time_folder = "RLSolver-master/rlsolver/result/eeco_obj_vs_time_2"
    sampling_speed_folder = "RLSolver-master/rlsolver/result/eeco_sampling_speed_2"
    # plot_obj_vs_time(obj_vs_time_folder,smooth = True, window_size=3,max_time=250)
    plot_sampling_speed(sampling_speed_folder, max_time=75, window_size=2, xticks_stride=10, smooth=True)
    plot_loss(loss_folder, smooth=True, window_size=3, max_time=2500)
    plot_inference_obj_vs_time(inference_obj_vs_time_folder, smooth=True, window_size=5, max_time=300)


if __name__ == "__main__":
    run()
