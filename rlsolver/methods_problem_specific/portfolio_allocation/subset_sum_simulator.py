import csv
import torch as th
from evaluator import EncoderBase64

TEN = th.Tensor
DataDir = './data/subset_sum'
'''simulator'''


def read_amount(file_name: str):
    file_name = f"{DataDir}/{file_name}"

    if file_name[-4:] == '.csv':
        with open(file_name, 'r') as f:
            lines = f.readlines()[1:]  # 忽略标题行
            amount = [float(line.split(',')[1]) for line in lines]
    elif file_name[-4:] == '.npy':
        import numpy as np
        amount = np.load(file_name)
    else:
        raise ValueError(f"| read_amount() file_name should be csv or npy, but {file_name}")

    amount = th.tensor(amount, dtype=th.float64)
    amount = (amount * 100).long()
    return amount


class SimulatorSubsetSum:
    def __init__(self, sim_name: str, device=th.device('cpu')):
        self.sim_name = sim_name
        self.device = device

        """data"""
        amount = read_amount(sim_name)
        self.num_nodes = amount.shape[0]

        '''data:amount'''
        self.amount = amount.to(device)

        '''data:lambda'''
        self.lamb = th.tensor([1, -1], dtype=th.float32, device=device)[None, :]

    def calculate_obj_values(self, xs: TEN, if_sum: bool = True) -> TEN:
        value1 = xs.long().sum(dim=1)
        const1 = th.abs((self.amount[None, :] * xs).sum(dim=1))

        value = th.stack((value1, const1), dim=1) * self.lamb
        return value.sum(dim=1) if if_sum else value

    def generate_xs_randomly(self, num_sims):
        xs = th.randint(0, 2, size=(num_sims, self.num_nodes), dtype=th.bool, device=self.device)
        return xs


def read_amount_tag(file_name: str):
    file_name = f"{DataDir}/{file_name}"

    with open(file_name, 'r') as f:
        lines = f.readlines()[1:]  # 忽略标题行
        lines = [line.split(',') for line in lines]

    amount = [float(line[1]) for line in lines]
    amount = th.tensor(amount, dtype=th.float64)
    amount = (amount * 100).long()

    tag = [str(line[2][:-1]) for line in lines]  # todo
    return amount, tag


class SimulatorSubsetSumWithTag(SimulatorSubsetSum):
    def __init__(self, sim_name: str, device=th.device('cpu')):
        super().__init__(sim_name, device)
        self.sim_name = sim_name
        self.device = device

        """data"""
        amount, tag = read_amount_tag(sim_name)
        self.num_nodes = amount.shape[0]

        '''data:amount'''
        self.amount = amount.to(device)

        '''data:tag'''
        default_value = 0
        tag_dict = {'JF': 1, 'JW': 2}
        tag = [tag_dict.get(key, default_value) for key in tag]
        tag = th.tensor(tag, dtype=th.int32)
        self.tag_00 = tag.eq(default_value).to(device)
        self.tag_jf = tag.eq(tag_dict['JF']).to(device)
        self.tag_jw = tag.eq(tag_dict['JW']).to(device)

        '''data:lambda'''
        self.lamb = th.tensor([1, -1, -1, -self.num_nodes], dtype=th.float32, device=device)[None, :]

    def calculate_obj_values(self, xs: TEN, if_sum: bool = True) -> TEN:
        value1 = xs.long().sum(dim=1)  # 尽可能选中更多的订单

        const1 = th.abs((self.amount[None, :] * xs).sum(dim=1))  # 被选中的订单包含买卖两个方向，求和后的资金量要尽可能接近0

        tag_jf_num = (self.tag_jf * xs).sum(dim=1)  # 被选中的JF订单数量
        tag_jw_num = (self.tag_jw * xs).sum(dim=1)  # 被选中的JW订单数量
        const2 = th.min(tag_jf_num, tag_jw_num)  # 订单数量少的标签，尽可能接近0

        tag_00_num = (self.tag_00 * xs).sum(dim=1)  # 被选中的非JF与非JW订单数量
        const3 = tag_00_num.eq(0)  # 被选中的非JF与非JW订单数量不能为0

        value = th.stack((value1, const1, const2, const3), dim=1) * self.lamb
        return value.sum(dim=1) if if_sum else value


def check_simulator():
    gpu_id = -1
    num_sims = 16
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    sim_name = 'xxx.csv'
    sim = SimulatorSubsetSum(sim_name=sim_name, device=device)
    for i in range(8):
        xs = sim.generate_xs_randomly(num_sims=num_sims)
        obj = sim.calculate_obj_values(xs=xs)
        print(f"| {i}  max_obj_value {obj.max().item()}")
    pass

    sim_name = 'xxx.csv'
    sim = SimulatorSubsetSumWithTag(sim_name=sim_name, device=device)
    for i in range(8):
        xs = sim.generate_xs_randomly(num_sims=num_sims)
        obj = sim.calculate_obj_values(xs=xs)
        print(f"| {i}  max_obj_value {obj.max().item()}")
    pass


def show_str_of_best_x_then_save_csv(sim: SimulatorSubsetSum, x_str: str) -> str:
    enc = EncoderBase64(num_nodes=sim.num_nodes)

    x = enc.str_to_bool(x_str)
    value_and_const = sim.calculate_obj_values(xs=x[None, :], if_sum=False)[0].cpu().data.numpy()
    value_rate = value_and_const[0] / sim.num_nodes

    read_csv = f"{DataDir}/{sim.sim_name}"
    with open(read_csv, 'r') as file:
        lines = file.readlines()

    # 添加新的一列
    new_column = ['solution_x', ] + [int(item) for item in x]
    new_lines = [line.strip() + f",{new_column[i]}" for i, line in enumerate(lines)]
    write_csv = f"{DataDir}/solution_x_{sim.sim_name}"
    with open(write_csv, 'w') as file:
        file.write('\n'.join(new_lines))
    return f"save {sim.sim_name:24}  | rate {value_rate:5.2f} | [value, constraint, ...]  {value_and_const}  "


def check_x():
    print('\ncheck x in SimulatorSubsetSum (pure)')




if __name__ == '__main__':
    # check_simulator()
    # check_x_in_simulator_subset_sum()
    check_x()
