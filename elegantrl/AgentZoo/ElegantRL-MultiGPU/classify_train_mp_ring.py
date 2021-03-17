import os
import time
import numpy as np
import numpy.random as rd

import torch
from Demo_deep_learning.yonv_utils import load_data_ary
from Demo_deep_learning.yonv_utils import load_torch_model
from Demo_deep_learning.yonv_utils import whether_remove_history
from torch.nn.utils import clip_grad_norm_

"""Github: Yonv1943 
======================================================
dataset    net        para(MB)   time(s)    accuracy
----------|----------|----------|----------|----------
MNIST      Conv2dNet  1.55       103        >0.99
MNIST      SE2dNet    1.57       108        >0.99
Fashion    Conv2dNet  1.55       103        >0.92
Fashion    SE2dNet    1.57       108        >0.93
CIFAR10    Conv2dNet  2.07       103        >0.79
CIFAR10    SE2dNet    2.09       108        >0.80
======================================================


Without evaluate
Type                    UsedTime
train_and_evaluate()    100%
train_and_evaluate_mp() 106%

"""


class Arguments:
    def __init__(self, ):
        self.train_epochs = [max(int(8 * 0.5 ** (i / 2)), 1) for i in range(8)]
        self.batch_sizes = [int(128 * 2 ** (i / 2)) for i in range(8)]
        self.mid_dim = 2 ** 8
        self.if_amp = False
        self.if_one_hot = True
        self.num_worker = 1

        self.mod_dir = 'tutorial_cnn'
        self.gpu_id = 0

        self.show_gap = 2 ** 1
        self.eval_gap = 2 ** 3

        self.net_class = None
        # self.data_path = '/mnt/sdb1/Yonv/datasets/Data/MNIST/MNIST.npz'
        self.data_path = '/mnt/sdb1/Yonv/datasets/Data/FashionMNIST/FashionMNIST.npz'
        self.img_shape = (28, 28, 1)
        # self.data_path = '/mnt/sdb1/Yonv/datasets/Data/CIFAR10/CIFAR10.npz'
        # self.img_shape = (32, 32, 3)

    def init_before_training(self, if_main=True):
        if if_main:
            whether_remove_history(self.mod_dir, remove=True)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        np.random.seed(1943 + int(time.time()))
        torch.manual_seed(1943 + rd.randint(0, int(time.time())))


'''single processing'''


def train_and_evaluate(args):
    net_class = args.net_class
    data_path = args.data_path
    img_shape = args.img_shape

    if_amp = args.if_amp
    if_one_hot = args.if_one_hot

    mid_dim = args.mid_dim
    mod_dir = args.mod_dir
    gpu_id = args.gpu_id
    train_epochs = args.train_epochs
    batch_sizes = args.batch_sizes
    show_gap = args.show_gap
    eval_gap = args.eval_gap
    del args

    whether_remove_history(mod_dir, remove=True)

    '''init env'''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # choose GPU:0
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    np.random.seed(1943 + int(time.time()))
    torch.manual_seed(1943 + rd.randint(0, int(time.time())))

    '''load data'''
    train_imgs, train_labs, eval__imgs, eval__labs = load_data_ary(data_path, img_shape, if_one_hot)
    train_imgs = torch.as_tensor(train_imgs, dtype=torch.float32, device=device)
    eval__imgs = torch.as_tensor(eval__imgs, dtype=torch.float32, device=device)

    label_data_type = torch.float32 if if_one_hot else torch.long
    train_labs = torch.as_tensor(train_labs, dtype=label_data_type, device=device)
    eval__labs = torch.as_tensor(eval__labs, dtype=label_data_type, device=device)
    del label_data_type
    train_len = train_imgs.shape[0]
    eval__len = eval__imgs.shape[0]
    eval_size = min(2 ** 12, eval__len)

    '''train model'''
    model = net_class(mid_dim, img_shape).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    from torch.nn import functional
    criterion = torch.nn.SmoothL1Loss() if if_one_hot else functional.nll_loss
    amp_scale = torch.cuda.amp.GradScaler()

    '''evaluator'''
    evaluator = Evaluator(eval__imgs, eval__labs, eval_size, eval_gap, show_gap, criterion)
    save_path = f'{mod_dir}/net.pth'

    '''if_amp'''

    def gradient_decent_original():
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

    def gradient_decent_amp():  # automatic mixed precision
        optimizer.zero_grad()
        amp_scale.scale(loss).backward()  # loss.backward()
        amp_scale.unscale_(optimizer)  # amp, clip_grad_norm_
        clip_grad_norm_(model.parameters(), max_norm=3.0)  # amp, clip_grad_norm_
        amp_scale.step(optimizer)  # optimizer.step()
        amp_scale.update()  # optimizer.step()

    gradient_decent = gradient_decent_amp if if_amp else gradient_decent_original

    print("Train Loop:")
    learning_rates = [1e-3, ] * len(train_epochs) + [1e-4, ] * 2 + [1e-5, ] * 2
    train_epochs.extend(train_epochs[-1:] * (len(learning_rates) - len(train_epochs)))
    batch_sizes.extend(batch_sizes[-1:] * (len(learning_rates) - len(batch_sizes)))

    for train_epoch, batch_size, learning_rate in zip(train_epochs, batch_sizes, learning_rates):
        optimizer.param_groups[0]['lr'] = learning_rate
        for epoch in range(train_epoch):
            loss_sum = 0
            model.train()

            '''train_it'''
            train_time = int(train_len / batch_size)
            for i in range(train_time):
                ids = rd.randint(train_len, size=batch_size)
                inp = train_imgs[ids]
                lab = train_labs[ids]

                out = model(inp)

                loss = criterion(torch.softmax(out, dim=1), lab)
                gradient_decent()
                loss_sum += loss.item()
            loss_avg = loss_sum / train_time

            evaluator.evaluate(model, batch_size, train_epoch, epoch, loss_avg)
    evaluator.final_print()

    torch.save(model.state_dict(), save_path)
    file_size = os.path.getsize(save_path) / (2 ** 20)  # Byte --> KB --> MB
    print(f"\nSave: {mod_dir} | {file_size:.2f} MB")


'''multiple processing ring'''


def train_and_evaluate_mp_ring(args):
    num_worker = args.num_worker
    import multiprocessing as mp

    process_list = list()

    pipe_eva1, pipe_eva2 = mp.Pipe()
    process_list.append(mp.Process(target=mp_evaluate, args=(args, pipe_eva1)))

    pipe_net_l1 = list()
    pipe_net_l2 = list()
    for _ in range(num_worker):
        pipe_net1, pipe_net2 = mp.Pipe()
        pipe_net_l1.append(pipe_net1)
        pipe_net_l2.append(pipe_net2)

    queue_data_l = list()
    for idx in range(num_worker):
        queue_data = mp.Queue(8)
        queue_data_l.append(queue_data)

        process_list.extend([mp.Process(target=mp_train, args=(args, idx, queue_data,
                                                               pipe_eva2, pipe_net_l1, pipe_net_l2[idx])),
                             mp.Process(target=mp_data, args=(args, idx, queue_data))])

    [p.start() for p in process_list]
    process_list[0].join()
    for pipe in [pipe_eva1, pipe_eva2] + pipe_net_l1 + pipe_net_l2:
        while pipe.poll():
            pipe.recv()
    for queue in queue_data_l:
        while queue.qsize() > 0:
            queue.get()
    [p.terminate() for p in process_list[1:]]


def mp_evaluate(args, pipe_eva1):
    args.init_before_training(if_main=True)

    net_class = args.net_class
    data_path = args.data_path
    img_shape = args.img_shape

    if_one_hot = args.if_one_hot
    num_worker = args.num_worker

    mid_dim = args.mid_dim
    mod_dir = args.mod_dir
    train_epochs = args.train_epochs
    batch_sizes = args.batch_sizes
    show_gap = args.show_gap
    eval_gap = args.eval_gap
    del args

    device = torch.device('cpu')
    '''load data'''
    train_imgs, train_labs, eval__imgs, eval__labs = load_data_ary(data_path, img_shape, if_one_hot)
    # train_imgs = torch.as_tensor(train_imgs, dtype=torch.float32, device=device)
    eval__imgs = torch.as_tensor(eval__imgs, dtype=torch.float32, device=device)

    label_data_type = torch.float32 if if_one_hot else torch.long
    # train_labs = torch.as_tensor(train_labs, dtype=label_data_type, device=device)
    eval__labs = torch.as_tensor(eval__labs, dtype=label_data_type, device=device)
    del label_data_type

    # train_len = train_imgs.shape[0]
    eval__len = eval__imgs.shape[0]
    eval_size = min(2 ** 12, eval__len)
    del train_imgs, train_labs

    '''train model'''
    model = net_class(mid_dim, img_shape).to(device)
    model_cpu = model.to(torch.device("cpu"))  # for pipe1_eva
    [setattr(param, 'requires_grad', False) for param in model_cpu.parameters()]
    del model

    for _ in range(num_worker):
        pipe_eva1.send(model_cpu.state_dict())
        # model_cpu_state_dict = pipe_eva2.recv()

    from torch.nn import functional
    criterion = torch.nn.SmoothL1Loss() if if_one_hot else functional.nll_loss

    '''init evaluate'''
    evaluator = Evaluator(eval__imgs, eval__labs, eval_size, eval_gap, show_gap, criterion)
    save_path = f'{mod_dir}/net.pth'

    learning_rates = [1e-3, ] * len(train_epochs) + [1e-4, ] * 2 + [1e-5, ] * 2
    train_epochs.extend(train_epochs[-1:] * (len(learning_rates) - len(train_epochs)))
    batch_sizes.extend(batch_sizes[-1:] * (len(learning_rates) - len(batch_sizes)))

    # pipe_eva2.send((idx, model_dict, batch_size, train_epoch, epoch, loss_avg))
    # pipe_eva2.send('break')
    pipe_receive = pipe_eva1.recv()

    print("Train Loop:")
    with torch.no_grad():
        while True:
            while pipe_eva1.poll():
                # pipe_eva2.send((idx, model_dict, batch_size, train_epoch, epoch, loss_avg))
                # pipe_eva2.send('break')
                pipe_receive = pipe_eva1.recv()
                if pipe_receive == 'break':
                    break
            if pipe_receive == 'break':
                break

            idx, model_dict, batch_size, train_epoch, epoch, loss_avg = pipe_receive
            model_cpu.load_state_dict(model_dict)

            evaluator.evaluate(model_cpu, batch_size, train_epoch, epoch, loss_avg)
    evaluator.final_print()

    torch.save(model_cpu.state_dict(), save_path)
    file_size = os.path.getsize(save_path) / (2 ** 20)  # Byte --> KB --> MB
    print(f"\nSave: {mod_dir} | {file_size:.2f} MB")


def mp_train(args, idx, queue_data, pipe_eva2, pipe_net_l1, pipe_net2):
    args.init_before_training(if_main=False)

    net_class = args.net_class
    # data_path = args.data_path
    img_shape = args.img_shape

    if_amp = args.if_amp
    if_one_hot = args.if_one_hot
    num_worker = args.num_worker

    mid_dim = args.mid_dim
    train_epochs = args.train_epochs
    batch_sizes = args.batch_sizes
    del args

    '''train model'''
    device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else 'cpu')
    model = net_class(mid_dim, img_shape).to(device)
    # pipe_eva1.send(model_cpu.state_dict())
    model_cpu_state_dict = pipe_eva2.recv()
    model.load_state_dict(model_cpu_state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    from torch.nn import functional
    criterion = torch.nn.SmoothL1Loss() if if_one_hot else functional.nll_loss
    amp_scale = torch.cuda.amp.GradScaler()

    # queue_data.put(train_len)
    train_len = queue_data.get()

    '''if_amp'''

    def gradient_decent_original():
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

    def gradient_decent_amp():  # automatic mixed precision
        optimizer.zero_grad()
        amp_scale.scale(loss).backward()  # loss.backward()
        amp_scale.unscale_(optimizer)  # amp, clip_grad_norm_
        clip_grad_norm_(model.parameters(), max_norm=3.0)  # amp, clip_grad_norm_
        amp_scale.step(optimizer)  # optimizer.step()
        amp_scale.update()  # optimizer.step()

    gradient_decent = gradient_decent_amp if if_amp else gradient_decent_original

    '''training loop'''
    learning_rates = [1e-3, ] * len(train_epochs) + [1e-4, ] * 2 + [1e-5, ] * 2
    train_epochs.extend(train_epochs[-1:] * (len(learning_rates) - len(train_epochs)))
    batch_sizes.extend(batch_sizes[-1:] * (len(learning_rates) - len(batch_sizes)))

    pipe_eva2_counter = idx

    for train_epoch, batch_size, learning_rate in zip(train_epochs, batch_sizes, learning_rates):
        optimizer.param_groups[0]['lr'] = learning_rate
        for epoch in range(train_epoch):
            loss_sum = 0
            model.train()

            '''train_it'''
            train_time = int(train_len / batch_size)
            for i in range(train_time):
                # queue_data.put((inp, lab))
                inp, lab = queue_data.get()

                out = model(inp)

                loss = criterion(torch.softmax(out, dim=1), lab)
                gradient_decent()
                loss_sum += loss.item()

                pipe_net_l1[(idx + 1) % num_worker].send(model)
                current_model = pipe_net2.recv()
                soft_update(model, current_model.to(device), tau=0.5)

            loss_avg = loss_sum / train_time

            pipe_eva2_counter = (pipe_eva2_counter + 1) % num_worker
            if pipe_eva2_counter == 0:
                model_cpu = model.state_dict()
                pipe_eva2.send((idx, model_cpu, batch_size, train_epoch, epoch, loss_avg))
                # pipe_receive = pipe_eva1.recv()

    pipe_eva2.send('break')
    # pipe_receive = pipe_eva1.recv()

    while True:
        time.sleep(4)


def mp_data(args, idx, queue_data):
    args.init_before_training(if_main=False)

    data_path = args.data_path
    img_shape = args.img_shape

    if_one_hot = args.if_one_hot

    batch_sizes = args.batch_sizes
    train_epochs = args.train_epochs
    del args

    device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else 'cpu')

    '''load data'''
    train_imgs, train_labs, eval__imgs, eval__labs = load_data_ary(data_path, img_shape, if_one_hot)
    train_imgs = torch.as_tensor(train_imgs, dtype=torch.float32, device=device)
    # eval__imgs = torch.as_tensor(eval__imgs, dtype=torch.float32, device=device)

    label_data_type = torch.float32 if if_one_hot else torch.long
    train_labs = torch.as_tensor(train_labs, dtype=label_data_type, device=device)
    # eval__labs = torch.as_tensor(eval__labs, dtype=label_data_type, device=device)
    del label_data_type

    train_len = train_imgs.shape[0]
    # eval__len = eval__imgs.shape[0]
    # eval_size = min(2 ** 12, eval__len)
    del eval__imgs, eval__labs

    queue_data.put(train_len)
    # train_len = queue_data.get()

    '''training loop'''
    learning_rates = [1e-3, ] * len(train_epochs) + [1e-4, ] * 2 + [1e-5, ] * 2
    train_epochs.extend(train_epochs[-1:] * (len(learning_rates) - len(train_epochs)))
    batch_sizes.extend(batch_sizes[-1:] * (len(learning_rates) - len(batch_sizes)))

    for train_epoch, batch_size, learning_rate in zip(train_epochs, batch_sizes, learning_rates):
        for epoch in range(train_epoch):
            train_time = int(train_len / batch_size)
            for i in range(train_time):
                ids = rd.randint(train_len, size=batch_size)
                inp = train_imgs[ids]
                lab = train_labs[ids]

                queue_data.put((inp, lab))
                # inp, lab = queue_data.get()

    while True:
        time.sleep(4)


'''Utils'''


class Evaluator:
    def __init__(self, eval__imgs, eval__labs, eval_size, eval_gap, show_gap, criterion):
        self.show_gap = show_gap
        self.eval__imgs = eval__imgs
        self.eval__labs = eval__labs
        self.eval__len = len(eval__labs)
        self.eval_gap = eval_gap
        self.eval_size = eval_size
        self.criterion = criterion

        self.start_time = time.time()
        self.eval_time = self.show_time = 0

        self.loss_avg = 0
        self.loss_eva = 0
        self.accuracy = 0

    def evaluate(self, model, batch_size, train_epoch, epoch, loss_avg):
        self.loss_avg = loss_avg
        time0 = time.time()
        if time0 - self.show_time > self.show_gap:
            self.show_time = time.time()
            print(f"|{batch_size:>4}/{train_epoch - epoch:>4} |loss {self.loss_avg:.4f}")

        if time0 - self.eval_time > self.eval_gap:
            '''evaluate_it'''
            self.eval_time = time.time()
            model.eval()
            loss_sum_eva = 0
            correct = 0

            eval__time = int(self.eval__len / self.eval_size)
            eval__time = eval__time + 1 if self.eval__len % self.eval_size else eval__time
            with torch.no_grad():
                for i in range(eval__time):
                    j = i * self.eval_size
                    inp = self.eval__imgs[j:j + self.eval_size]
                    lab = self.eval__labs[j:j + self.eval_size]
                    out = model(inp)

                    loss_sum_eva += self.criterion(
                        torch.softmax(out, dim=1), lab).item() * lab.shape[0]

                    predict = out.argmax(dim=1, keepdim=True)
                    int_lab = lab.argmax(dim=1, keepdim=True)
                    predict_bool = predict.eq(int_lab.view_as(predict))
                    correct += predict_bool.sum().item()

            self.loss_eva = loss_sum_eva / self.eval__len
            self.accuracy = correct / self.eval__len
            print(f"|{batch_size:>4}/{train_epoch:>4} |loss {self.loss_avg:.4f} "
                  f"|EvaLoss {self.loss_eva:.4f} |Accu {self.accuracy:.4f}")

    def final_print(self):
        print(f"TimeUsed: {time.time() - self.start_time:4.0f}|loss {self.loss_avg:.4f}    "
              f"| EvaLoss {self.loss_eva:.4f}   |Accu {self.accuracy:.4f}")


def soft_update(target_net, current_net, tau):
    for tar, cur in zip(target_net.parameters(), current_net.parameters()):
        tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))


'''run'''


def run_test(args):
    mod_dir = args.mod_dir
    gpu_id = args.gpu_id
    net_class = args.net_class
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # choose GPU:0
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model = net_class().to(device)
    load_torch_model(mod_dir, model)

    in_put = np.zeros((1, 1, 28, 28), dtype=np.float32)
    in_put = torch.tensor(in_put, dtype=torch.float32, device=device)
    output = model(in_put)
    output = output.cpu().data.numpy()[0]
    print(np.argmax(output), output)


def run_main():
    args = Arguments()
    args.gpu_id = '0, 1, 2, 3'
    args.if_amp = False
    args.if_one_hot = True
    args.num_worker = 4
    # from Demo_deep_learning.classify_network import Res1dNet
    # from Demo_deep_learning.classify_network import Conv2dNet
    # from Demo_deep_learning.classify_network import SE2dNet
    from Demo_deep_learning.classify_network import ConvNet
    args.net_class = ConvNet

    # args.train_epochs = [max(int(8 * 0.5 ** (i / 2)), 1) for i in range(8)]
    args.train_epochs = [max(int(0 * 0.5 ** (i / 2)), 1) for i in range(8)]
    args.batch_sizes = [int(128 * 2 ** (i / 2)) for i in range(8)]
    args.data_path = '/mnt/sdb1/Yonv/datasets/Data/CIFAR10/CIFAR10.npz'
    args.img_shape = (32, 32, 3)

    # train_and_evaluate(args)
    # train_and_evaluate_mp(args)
    train_and_evaluate_mp_ring(args)
    # run_test(args.mod_dir, args.gpu_id)


if __name__ == '__main__':
    run_main()
