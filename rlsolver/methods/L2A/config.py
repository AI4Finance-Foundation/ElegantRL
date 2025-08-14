import os
from rlsolver.methods.util_read_data import GraphList, obtain_num_nodes

ModelDir = './model'  # FIXME plan to cancel


class ConfigGraph:
    def __init__(self, graph_list: GraphList = None, graph_type: str = 'max_cut', num_nodes: int = 0):
        num_nodes = num_nodes if num_nodes > 0 else obtain_num_nodes(graph_list=graph_list)

        self.graph_type = graph_type
        self.num_nodes = num_nodes

        '''train'''
        self.num_buffers = 4  # 训练图向量需要的数据集个数
        self.buffer_size = 2 ** 12  # 每个数据集里预先生成的图的数量
        self.buffer_repeats = 128  # 数据重复使用的次数
        self.buffer_dir = './buffer'  # 存放缓存数据的文件夹，里面可以存放为了训练预先生成的随机图邻接bool矩阵

        self.batch_size = 2 ** 5  # 训练的并行图的数量
        self.train_times = 2 ** 4  # 训练的次数
        self.weight_decay = 0  # 2 ** -16  # 优化器的权重衰减
        self.learning_rate = 2 ** -14  # 优化器的学习率
        self.show_gap = 2 ** 5  # 训练时，打印训练进度的间隔步数

        '''model'''
        self.num_heads = 2
        self.num_layers = 3
        self.mid_dim = 32
        self.inp_dim = num_nodes  # 输入是邻接矩阵
        self.out_dim = num_nodes * 3  # 输出是2+1幅热图

        from math import log2
        sqrt_num_nodes = int(log2(num_nodes))
        self.embed_dim = max(sqrt_num_nodes - sqrt_num_nodes % self.num_heads, 16)  # 编码后的节点嵌入向量的长度


class ConfigPolicy:
    def __init__(self, graph_list: GraphList = None, graph_type: str = 'max_cut', num_nodes: int = 0):
        num_nodes = num_nodes if num_nodes > 0 else obtain_num_nodes(graph_list=graph_list)

        self.graph_type = graph_type
        self.num_nodes = num_nodes

        '''train'''
        self.num_sims = 2 ** 6  # LocalSearch 的初始解数量
        self.num_repeats = 2 ** 5  # LocalSearch 对于每以个初始解进行复制的数量
        self.num_searches = 2 ** 2  # LocalSearch 添加噪声的次数
        self.top_k = num_nodes // 4  # LocalSearch 搜索不确定性排名靠前的k个比特位
        self.reset_gap = 2 ** 0  # 重置并开始新的搜索需要的迭代步数
        self.num_iters = 2 ** 8  # 进行迭代搜索的总步数
        self.num_sgd_steps = 2 ** 2  # 每一次根据监督信号进行梯度下降的次数
        self.lambda_entropy = 4  # PPO 策略熵的权重（退火方案会控制策略熵，在一个周期内由entropy_weight*1变化到0）
        self.repeat_times = 8  # PPO 在on-policy训练中，数据重复使用的次数
        self.clip_ratio = 0.25  # PPO 重要性采样中，梯度裁剪的比率

        self.learning_rate = 2 ** -16  # 优化器的学习率
        self.weight_decay = 0  # 2 ** -16 # 优化器的权重衰减
        self.net_path = f"{ModelDir}/policy_net_{graph_type}_Node{num_nodes}.pth"  # policy_net的保存路径

        self.show_gap = 2 ** 2  # 训练时，打印训练进度的间隔步数

        '''model'''
        self.num_heads = 2
        self.num_layers = 1
        self.seq_len = 16
        self.mid_dim = 32
        self.inp_dim = num_nodes  # 输入是邻接矩阵
        self.out_dim = 2  # 输出是节点对应的概率

        from math import log2
        sqrt_num_nodes = int(log2(num_nodes))
        self.embed_dim = max(sqrt_num_nodes - sqrt_num_nodes % self.num_heads, 16)  # 编码后的节点嵌入向量的长度

    def load_net(self, net_path: str = '', device=None, if_valid: bool = False):
        import torch as th
        net_path = net_path if net_path else self.net_path
        device = device if device else th.device('cpu')

        # from network import PolicyRNN
        # net = PolicyRNN(inp_dim=self.inp_dim, mid_dim=self.mid_dim, out_dim=self.out_dim,
        #                 embed_dim=self.embed_dim, num_heads=self.num_heads, num_layers=self.num_layers).to(device)
        from network import PolicyTRS
        net = PolicyTRS(inp_dim=self.inp_dim, mid_dim=self.mid_dim, out_dim=self.out_dim,
                        embed_dim=self.embed_dim, num_heads=self.num_heads, num_layers=self.num_layers).to(device)
        if if_valid:
            if not not os.path.isfile(net_path):
                raise FileNotFoundError(f"| ConfigPolicy.load_net()  net_path {net_path}")
            net.load_state_dict(th.load(net_path, map_location=device))
        else:  # if_train
            pass
        return net
