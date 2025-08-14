import sys
import torch as th
from math import log10 as math_log10

TEN = th.Tensor


def get_nodes_list_and_band_edges_of_tensor_train(len_list: int = 4):
    nodes = [[] for _ in range(len_list)]  # 初始化邻接表
    for i in range(len_list):
        if i > 0:
            nodes[i].append(i - 1)
        if i < len_list - 1:
            nodes[i].append(i + 1)
        nodes[i].append(i + len_list)
        nodes.append([i])
    return nodes

def get_nodes_list_and_band_edges_of_tensor_ring(len_list: int = 4):
    ban_edges = len_list

    nodes = [[] for _ in range(len_list)]  # 初始化邻接表
    for i in range(len_list):
        nodes[i].append((i - 1) % len_list)
        nodes[i].append((i + 1) % len_list)

        nodes[i].append(i + len_list)
        nodes.append([i])
    return nodes, ban_edges



def get_nodes_list_and_ban_edges_of_tensor_tree(depth: int = 3):
    depth -= 1  # todo standard depth for tensor tree

    # 初始化二叉树的二维列表
    num_nodes = 2 ** (depth + 1) - 1
    ban_edges = 2 ** depth
    tree = [[] for i in range(num_nodes)]

    # 添加二叉树的边
    def add_edges(_depth, node=0, org_node=-1):
        left_node = node * 2 + 1
        right_node = node * 2 + 2

        tree[node].append(org_node) if org_node >= 0 else None
        if _depth == 0:
            return
        tree[node].append(left_node)
        tree[node].append(right_node)

        org_node = node
        _depth -= 1
        add_edges(_depth, left_node, org_node)
        add_edges(_depth, right_node, org_node)

    add_edges(depth)
    return tree, ban_edges


def get_nodes_ary(nodes_list: list) -> TEN:
    # nodes_list = NodesSycamore
    nodes_ary = th.zeros((len(nodes_list), max([len(nodes) for nodes in nodes_list])), dtype=th.int) - 1
    # # -1 表示这里没有连接
    for i, nodes in enumerate(nodes_list):
        for j, node in enumerate(nodes):
            nodes_ary[i, j] = node
    return nodes_ary


def get_edges_ary(nodes_ary: TEN) -> TEN:
    edges_ary = th.zeros_like(nodes_ary, dtype=nodes_ary.dtype)
    edges_ary[nodes_ary >= 0] = -2  # -2 表示这里的 edge_i 需要被重新赋值
    edges_ary[nodes_ary == -1] = -1  # -1 表示这里的 node 没有连接另一个 node

    num_edges = 0
    '''get nodes_ary'''
    # for i, nodes in enumerate(nodes_ary):  # i 表示节点的编号
    #     for j, node in enumerate(nodes):  # node 表示跟编号为i的节点相连的另一个节点
    #         edge_i = edges_ary[i, j]
    #         if edge_i == -2:
    #             _j = th.where(nodes_ary[node] == i)
    #             edges_ary[i, j] = num_edges
    #             edges_ary[node, _j] = num_edges
    #             num_edges += 1
    '''get nodes_ary and sort the ban edges to large indices'''
    for i, nodes in list(enumerate(nodes_ary))[::-1]:  # i 表示节点的编号
        for j, node in enumerate(nodes):  # node 表示跟编号为i的节点相连的另一个节点
            edge_i = edges_ary[i, j]
            if edge_i == -2:
                nodes_ary_node: TEN = nodes_ary[node]
                _j = th.where(nodes_ary_node == i)

                edges_ary[i, j] = num_edges
                edges_ary[node, _j] = num_edges
                num_edges += 1
    _edges_ary = edges_ary.max() - edges_ary
    _edges_ary[edges_ary == -1] = -1
    edges_ary = _edges_ary
    return edges_ary


def get_node_dims_arys(nodes_ary: TEN) -> list:
    num_nodes = nodes_ary.shape[0]

    arys = []
    for nodes in nodes_ary:
        positive_nodes = nodes[nodes >= 0].long()
        ary = th.zeros((num_nodes,), dtype=th.int)  # 因为都是2，所以我用0 表示 2**0==1
        ary[positive_nodes] = 1  # 2量子比特门，这里的计算会带来2个单位的乘法，因为都是2，所以我用1 表示 2**1==2
        arys.append(ary)
    return arys


def get_node_bool_arys(nodes_ary: TEN) -> list:
    num_nodes = nodes_ary.shape[0]

    arys = []
    for i, nodes in enumerate(nodes_ary):
        ary = th.zeros((num_nodes,), dtype=th.bool)
        ary[i] = True
        arys.append(ary)
    return arys


class TensorNetworkEnv:
    def __init__(self, nodes_list: list, ban_edges: int, device: th.device, if_vec: bool = True):
        self.device = device

        '''build node_arys and edges_ary'''
        nodes_ary = get_nodes_ary(nodes_list)
        num_nodes = nodes_ary.max().item() + 1
        assert num_nodes == nodes_ary.shape[0]

        edges_ary = get_edges_ary(nodes_ary)
        num_edges = edges_ary.max().item() + 1
        assert num_edges == (edges_ary != -1).sum() / 2

        self.nodes_ary = nodes_ary
        self.edges_ary = edges_ary.to(device)
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.ban_edges = ban_edges

        '''build for get_log10_multiple_times'''
        node_dims_arys = get_node_dims_arys(nodes_ary)
        assert num_edges == sum([(ary == 1).sum().item() for ary in node_dims_arys]) / 2

        node_bool_arys = get_node_bool_arys(nodes_ary)
        assert num_nodes == sum([ary.sum() for ary in node_bool_arys])

        self.dims_ten = th.stack(node_dims_arys).type(th.float32).to(device)
        self.bool_ten = th.stack(node_bool_arys).type(th.bool).to(device)
        default_num_envs = 1
        self.dims_tens = th.stack([self.dims_ten.clone() for _ in range(default_num_envs)])
        self.bool_tens = th.stack([self.bool_ten.clone() for _ in range(default_num_envs)])

        self.update_pow_counts = self.update_pow_vectorized if if_vec else self.update_pow_vanilla

    def get_log10_multiple_times(self, edge_sorts: TEN, if_acc: bool = False) -> TEN:
        # edge_argsort = th.rand(self.num_edges).argsort()
        device = self.device
        edges_ary: TEN = self.edges_ary
        num_envs, run_edges = edge_sorts.shape

        if not (self.dims_tens.shape[0] == self.bool_tens.shape[0] == num_envs):
            self.dims_tens = th.stack([self.dims_ten.clone() for _ in range(num_envs)])
            self.bool_tens = th.stack([self.bool_ten.clone() for _ in range(num_envs)])
        dims_tens = self.dims_tens.clone()
        bool_tens = self.bool_tens.clone()

        pow_counts = th.zeros((num_envs, run_edges), dtype=th.float64, device=device)
        for i in range(run_edges):
            edge_is = edge_sorts[:, i]
            self.update_pow_counts(i, edge_is, edges_ary, dims_tens, bool_tens, pow_counts)

        result = self.get_multiple_times_vectorized(pow_counts) if if_acc \
            else self.get_multiple_times_accurately(pow_counts)
        return result.detach()

    @staticmethod
    def update_pow_vanilla(i: int, edge_is: TEN, edges_ary: TEN,
                           dims_tens: TEN, bool_tens: TEN, pow_counts: TEN):
        num_envs = pow_counts.shape[0]

        for j in range(num_envs):
            edge_i = edge_is[j]
            dims_arys = dims_tens[j]
            bool_arys = bool_tens[j]

            '''find two nodes of an edge_i'''
            node_i0, node_i1 = th.where(edges_ary == edge_i)[0]  # 找出这条edge 两端的node
            # assert isinstance(node_i0.item(), int)
            # assert isinstance(node_i1.item(), int)

            '''whether node_i0 and node_i1 are different'''
            if_diff = th.logical_not(bool_arys[node_i0, node_i1])

            '''calculate the multiple and avoid repeat'''
            ct_dims = dims_arys[node_i0] + dims_arys[node_i1] * if_diff  # 计算收缩后的node 的邻接张量的维数以及来源
            ct_bool = bool_arys[node_i0] | bool_arys[node_i1]  # 计算收缩后的node 由哪些原初node 合成
            # assert ct_dims.shape == (num_nodes, )
            # assert ct_bool.shape == (num_nodes, )

            # 收缩掉的edge 只需要算一遍乘法。因此上面对 两次重复的指数求和后乘以0.5
            pow_count = ct_dims.sum(dim=0) - (ct_dims * ct_bool).sum(dim=0) * 0.5
            pow_counts[j, i] = pow_count * if_diff

            '''adjust two list: dims_arys, bool_arys'''
            # 如果两个张量是一样的，那么 `ct_bool & if_diff` 就会全部变成 False，让下面这行代码不修改任何数值
            ct_dims[ct_bool & if_diff] = 0  # 把收缩掉的边的乘法数量赋值为2**0，接下来不再参与乘法次数的计算
            dims_tens[j, ct_bool] = ct_dims.repeat(1, 1)  # 根据 bool 将所有收缩后的节点都刷新成相同的信息
            bool_tens[j, ct_bool] = ct_bool.repeat(1, 1)  # 根据 bool 将所有收缩后的节点都刷新成相同的信息

    @staticmethod
    def update_pow_vectorized(i: int, edge_is: TEN, edges_ary: TEN,
                              dims_tens: TEN, bool_tens: TEN, pow_counts: TEN):
        num_envs = pow_counts.shape[0]
        env_is = th.arange(num_envs, device=pow_counts.device)

        '''find two nodes of an edge_i'''
        vec_edges_ary: TEN = edges_ary[None, :, :]
        vec_edges_is: TEN = edge_is[:, None, None]
        res = th.where(vec_edges_ary == vec_edges_is)[1]  # Find the two nodes on either end of this edge
        res = res.reshape((num_envs, 2))
        node_i0s, node_i1s = res[:, 0], res[:, 1]
        # assert node_i0s.shape == (num_envs, )
        # assert node_i1s.shape == (num_envs, )

        '''whether node_i0 and node_i1 are different'''
        if_diffs = th.logical_not(bool_tens[env_is, node_i0s, node_i1s])
        # assert if_diffs.shape == (num_envs, )

        '''calculate the multiple and avoid repeat'''
        ct_dimss = dims_tens[env_is, node_i0s] + dims_tens[env_is, node_i1s] * if_diffs.unsqueeze(1)
        # assert ct_dimss.shape == (num_envs, num_nodes)  # get dim and src of nodes after contraction
        ct_bools = bool_tens[env_is, node_i0s] | bool_tens[env_is, node_i1s]
        # assert ct_bools.shape == (num_envs, num_nodes)  # get which original nodes compose after contraction

        # The multiplication for the contracted edge only needs to be computed once.
        # Therefore, the sum of the two repeated indices is added and then multiplied by 0.5.
        pow_count = ct_dimss.sum(dim=1) - (ct_dimss * ct_bools).sum(dim=1) * 0.5
        pow_counts[:, i] = pow_count * if_diffs

        '''adjust two list: dims_arys, bool_arys'''
        for j in range(num_envs):
            ct_dims = ct_dimss[j]  # get dim and src of nodes after contraction
            ct_bool = ct_bools[j]  # get which original nodes compose after contraction

            # If two tensors are the same, ct_bool & if_diff will become all False,
            # making the following line of code not modify any values.
            ct_dims[ct_bool & if_diffs[j]] = 0  # Set the mul to 2**0, it will not be included in following mul counts
            dims_tens[j, ct_bool] = ct_dims.repeat(1, 1)  # Refresh all contracted nodes with same info via to bool val
            bool_tens[j, ct_bool] = ct_bool.repeat(1, 1)  # Refresh all contracted nodes with same info via to bool val

    @staticmethod
    def get_multiple_times_accurately(pow_timess: TEN) -> TEN:
        num_envs = pow_timess.shape[0]
        # 缓慢但是完全不损失精度的计算方法
        multiple_times = []
        pow_timess = pow_timess.cpu().numpy()
        for env_id in range(num_envs):
            multiple_time = 0
            for pow_time in pow_timess[env_id, :]:
                multiple_time = multiple_time + 2 ** pow_time
            multiple_time = math_log10(multiple_time)
            multiple_times.append(multiple_time)
        return th.tensor(multiple_times, dtype=th.float64)

    @staticmethod
    def get_multiple_times_vectorized(pow_timess: TEN) -> TEN:
        device = pow_timess.device
        # 快速，但是有效数值有 1e-7 的计算方法（以下都是 float64）
        adj_pow_times = pow_timess.max(dim=1)[0] - 960  # automatically set `max - 960`, 960 < the limit 1024
        # 计算这个乘法个数时，即便用 float64 也偶尔会过拟合，所以先除以 2**temp_power ，求log10 后再恢复它
        multiple_times = (th.pow(2, pow_timess - adj_pow_times.unsqueeze(1))).sum(dim=1)
        multiple_times = multiple_times.log10() + adj_pow_times / th.log2(th.tensor((10,), device=device))
        # adj_pow_times / th.log2(th.tensor((10, ), device=device))  # Change of Base Formula
        return multiple_times

    def convert_edge_sort_to_node2s(self, edge_sort: TEN) -> list:
        edges_ary: TEN = self.edges_ary.cpu()
        edge_sort = edge_sort.cpu()

        run_edges = edge_sort.shape[0]
        assert run_edges == self.num_edges - self.ban_edges

        node2s = []
        for i in range(run_edges):
            edge_i = edge_sort[i]

            '''find two nodes of an edge_i'''
            node_i0, node_i1 = th.where(edges_ary == edge_i)[0]  # 找出这条edge 两端的node
            # assert isinstance(node_i0.item(), int)
            # assert isinstance(node_i1.item(), int)
            node2s.append((node_i0.item(), node_i1.item()))
        return node2s

    def convert_node2s_to_edge_sort(self, node2s: list) -> TEN:
        edges_ary: TEN = self.edges_ary.clone().cpu()
        # nodes_ary: TEN = self.nodes_ary.clone().cpu()

        edges_tmp = [set(edges.tolist()) for edges in edges_ary]
        for edges in edges_tmp:
            edges.discard(-1)

        edge_sort = []
        for node_i0, node_i1 in node2s:
            edge_i0_set = edges_tmp[node_i0]
            edge_i1_set = edges_tmp[node_i1]
            print(f"{node_i0:4} {str(edge_i0_set):17}    "
                  f"{node_i1:4} {str(edge_i1_set):17}    ")

            edge_is = edge_i0_set.intersection(edge_i1_set)
            edge_i = sorted(list(edge_is))[0]  # ordered
            #  edge_i = edge_is.pop()  # disordered
            edge_sort.append(edge_i)

            edge_01_set = edge_i0_set.union(edge_i1_set)
            edges_tmp[node_i0] = edge_01_set
            edges_tmp[node_i1] = edge_01_set

        edge_sort = th.tensor(edge_sort)
        return edge_sort


def convert_str_ary_to_list_as_edge_sort(str_ary: str) -> list:
    str_ary = str_ary.replace('   ', ',').replace('  ', ',').replace(' ', ',').replace('[,', '[ ')
    return eval(str_ary)


'''unit tests'''


def unit_test_get_log10_multiple_times():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    # nodes_list, ban_edges = NodesSycamoreN12M14, 0
    # nodes_list, ban_edges = NodesSycamoreN14M14, 0
    # nodes_list, ban_edges = NodesSycamoreN53M12, 0
    nodes_list, ban_edges = get_nodes_list_and_band_edges_of_tensor_train(len_list=8)
    # nodes_list, ban_edges = get_nodes_list_of_tensor_train(len_list=100), 100
    # nodes_list, ban_edges = get_nodes_list_of_tensor_train(len_list=2000), 2000
    # from TNCO_env import get_nodes_list_of_tensor_tree
    # nodes_list, ban_edges = get_nodes_list_of_tensor_tree(depth=3), 2 ** (3 - 1)

    env = TensorNetworkEnv(nodes_list=nodes_list, ban_edges=ban_edges, device=device)
    print(f"\nnum_nodes      {env.num_nodes:9}"
          f"\nnum_edges      {env.num_edges:9}"
          f"\nban_edges      {env.ban_edges:9}")
    num_envs = 6

    # th.save(edge_arys, 'temp.pth')
    # edge_arys = th.load('temp.pth', map_location=device)

    edge_arys = th.rand((num_envs, env.num_edges - env.ban_edges), device=device)
    multiple_times = env.get_log10_multiple_times(edge_sorts=edge_arys.argsort(dim=1))
    print(f"multiple_times(log10) {multiple_times.cpu().numpy()}")

    edge_arys = th.rand((num_envs, env.num_edges - env.ban_edges), device=device)
    multiple_times = env.get_log10_multiple_times(edge_sorts=edge_arys.argsort(dim=1), if_acc=True)
    print(f"multiple_times(log10) if_vec=True  {multiple_times.cpu().numpy()}")
    multiple_times = env.get_log10_multiple_times(edge_sorts=edge_arys.argsort(dim=1), if_acc=False)
    print(f"multiple_times(log10) if_vec=False {multiple_times.cpu().numpy()}")






def unit_test_warm_up():
    gpu_id = 0
    warm_up_size = 2 ** 8  # 2 ** 14
    target_score = -th.inf  # 7.0
    """
    你可以选择 target_score = 负无穷，然后设置你想要的 warm_up_size ，直接测试仿真环境 并行与否 的耗时
    也可以选择 warm_up_size = 正无穷，然后设置你想要的 target_score ，直接测试仿真环境 并行与否 的耗时
    """
    if_vec = True  # 设置 if_vec = True。可以在有GPU的情况下，高效快速地并行计算乘法次数
    if_acc = True  # 设置 if_acc = True。这种设置虽然慢，但是它使用int计算，能非常精确且不溢出地算出结果（能避免nan）
    env_nums = 2 ** 8
    """
    在启用 if_vec = True 的情况下，你可以设置 env_nums 来主动设置并行子环境数量。
    越好的GPU，就可以设置越大的 env_nums，设置到 GPU使用率无法再提高的情况下，会得到一个接近最高性价比的 env_nums
    """

    # nodes_list, ban_edges = NodesSycamoreN12M14, 0
    # nodes_list, ban_edges = get_nodes_list_and_band_edges_of_tensor_train(2000), 2000
    nodes_list, ban_edges = get_nodes_list_and_ban_edges_of_tensor_tree(depth=6)

    """
    这里你可以选择不同的张量电路无向图，在这里跑不同的任务。注意选择正确的 ban_edges 以及对应的 target_socre
    """

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    env = TensorNetworkEnv(nodes_list=nodes_list, ban_edges=ban_edges, device=device, if_vec=if_vec)
    dim = env.num_edges - env.ban_edges

    min_score = th.inf

    from time import time as timer
    start_time = timer()
    for i in range(warm_up_size // env_nums):
        thetas = th.rand((env_nums, dim), dtype=th.float32, device=device)
        thetas = ((thetas - thetas.mean(dim=1, keepdim=True)) / (thetas.std(dim=1, keepdim=True) + 1e-6))

        scores = env.get_log10_multiple_times(edge_sorts=thetas.argsort(dim=1), if_acc=if_acc)
        min_score = min(scores.min(dim=0)[0].item(), min_score)
        print(f"MinScore {min_score:16.9f}  UsedTime {timer() - start_time:9.3f}  SearchNum {(i + 1) * env_nums:9.0f}")
        if min_score < target_score:
            break
    """
    min_score 是这一次 warm_up 搜索到的最优分数
    UsedTime 是开始训练到此刻的耗时
    SearchNum 是开始训练到此刻的搜索次数

    实验结果：
    nodes_list, ban_edges = get_nodes_list_of_tensor_train(100), 100
    if_acc = False

    if_vec = False       MinScore 30.404030  UsedTime   151.614  SearchNum      4096
    env_nums = 2 ** 2    MinScore 30.404030  UsedTime    79.989  SearchNum      4096
    env_nums = 2 ** 4    MinScore 30.404030  UsedTime    55.738  SearchNum      4096
    env_nums = 2 ** 8    MinScore 30.404030  UsedTime    52.963  SearchNum      4096
    env_nums = 2 ** 12   MinScore 30.404030  UsedTime    52.031  SearchNum      4096
    """


if __name__ == '__main__':
    # unit_test_get_log10_multiple_times()
    # unit_test_convert_node2s_to_edge_sorts()
    # unit_test_convert_node2s_to_edge_sorts_of_load()
    # unit_test_edge_sorts_to_log10_multiple_times()
    unit_test_warm_up()
