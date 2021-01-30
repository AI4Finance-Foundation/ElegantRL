import os
import time
import numpy as np
import numpy.random as rd

import torch
from Tutorial.yonv_utils import load_data, load_data_ary
from Tutorial.yonv_utils import load_torch_model
from Tutorial.yonv_utils import whether_remove_history
# from classify_network import Res1dNet as Net
# from classify_network import Conv2dNet as Net
# from Tutorial.classify_network import SE2dNet as Net
from Tutorial.classify_network import ConvNet as Net
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

"""


class Arguments:
    def __init__(self, ):
        self.train_epochs = [max(int(8 * 0.5 ** (i / 2)), 1) for i in range(8)]
        self.batch_sizes = [int(128 * 2 ** (i / 2)) for i in range(8)]
        self.mid_dim = 2 ** 8

        self.mod_dir = 'tutorial_conv2d'
        self.gpu_id = 0

        self.show_gap = 2 ** 1
        self.eval_gap = 2 ** 3


def run_train__data_loader(args):
    # data_path = '/mnt/sdb1/Yonv/datasets/Data/MNIST/MNIST.npz'
    # data_path = '/mnt/sdb1/Yonv/datasets/Data/FashionMNIST/FashionMNIST.npz'
    # img_shape = (28, 28, 1)

    data_path = '/mnt/sdb1/Yonv/datasets/Data/CIFAR10/CIFAR10.npz'
    img_shape = (32, 32, 3)

    if_amp = True

    mid_dim = args.mid_dim
    mod_dir = args.mod_dir
    gpu_id = args.gpu_id
    train_times = args.train_epochs
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
    import torch.utils.data as data
    train_images, train_labels, eval_images, eval_labels = load_data(data_path, img_shape)

    train_sets = data.TensorDataset(train_images, train_labels)
    eval_sets = data.TensorDataset(eval_images, eval_labels)

    eval_loader = data.DataLoader(
        eval_sets, batch_size=batch_sizes[0], shuffle=False, drop_last=False, )
    del eval_sets

    '''train model'''
    from torch.nn import functional
    model = Net(mid_dim, img_shape).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = functional.nll_loss
    amp_scale = torch.cuda.amp.GradScaler()

    save_path = load_torch_model(mod_dir, model)

    print("Train Loop:")
    from time import time as timer
    start_time = timer()
    eval_time = show_time = 0
    loss_avg = loss_eva = accuracy = None

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

    learning_rates = [1e-3, ] * len(train_times) + [1e-4, ] * 2 + [1e-5, ] * 2
    train_times.extend(train_times[-1:] * (len(learning_rates) - len(train_times)))
    batch_sizes.extend(batch_sizes[-1:] * (len(learning_rates) - len(batch_sizes)))

    for train_time, batch_size, learning_rate in zip(train_times, batch_sizes, learning_rates):
        optimizer.param_groups[0]['lr'] = learning_rate
        for epoch in range(train_time):
            train_loader = data.DataLoader(train_sets, batch_size=batch_size, shuffle=True, drop_last=True)
            loss_sum = 0
            model.train()

            '''train_it'''
            for image, target in train_loader:
                image = image.to(device)
                target = target.to(device)

                output = model(image)
                loss = criterion(output, target)

                gradient_decent()

                loss_sum += loss.item()

            loss_avg = loss_sum / len(train_loader)

            time0 = timer()
            if time0 - show_time > show_gap:
                show_time = timer()
                print(f"|{batch_size:>4}/{train_time:>4}   |loss {loss_avg:.4f}")

            if time0 - eval_time > eval_gap:
                '''evaluate_it'''
                eval_time = timer()
                model.eval()
                loss_sum = 0
                correct = 0
                with torch.no_grad():
                    for image, target in eval_loader:
                        image = image.to(device)
                        # target_one_hot = torch.as_tensor(np.eye(10)[target], dtype=torch.float16, device=device)
                        target = target.to(device)

                        output = model(image)

                        loss_sum += criterion(output, target, reduction='mean').item()  # todo
                        # loss_sum += criterion(output, target_one_hot).mean().item()

                        predict = output.argmax(dim=1, keepdim=True)
                        predict_bool = predict.eq(target.view_as(predict))
                        correct += predict_bool.float().mean().item()

                eval_loader_len = len(eval_loader)
                loss_avg = loss_sum / eval_loader_len
                accuracy = correct / eval_loader_len
                print(f"|{batch_size:>4}/{train_time:>4}   |loss {loss_avg:.4f}   "
                      f"|EvaLoss {loss_avg:.4f}   |Accu {accuracy:.4f}")

    print(f"TimeUsed:{timer() - start_time:4.0f}|loss {loss_avg:.4f}   "
          f"| EvaLoss {loss_avg:.4f}   |Accu {accuracy:.4f}")

    torch.save(model.state_dict(), save_path)
    file_size = os.path.getsize(save_path) / (2 ** 20)  # Byte --> KB --> MB
    print(f"\nSave: {mod_dir} | {file_size:.2f} MB")


def run_train(args):
    # data_path = '/mnt/sdb1/Yonv/datasets/Data/MNIST/MNIST.npz'
    # data_path = '/mnt/sdb1/Yonv/datasets/Data/FashionMNIST/FashionMNIST.npz'
    # img_shape = (28, 28, 1)

    data_path = '/mnt/sdb1/Yonv/datasets/Data/CIFAR10/CIFAR10.npz'
    img_shape = (32, 32, 3)

    if_amp = False
    if_one_hot = True

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
    eval_size = min(2**12, eval__len)

    '''train model'''
    from torch.nn import functional

    model = Net(mid_dim, img_shape).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.SmoothL1Loss() if if_one_hot else functional.nll_loss
    amp_scale = torch.cuda.amp.GradScaler()

    save_path = load_torch_model(mod_dir, model)

    print("Train Loop:")
    from time import time as timer
    start_time = timer()
    eval_time = show_time = 0
    loss_avg = loss_eva = accuracy = None

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

            time0 = timer()
            if time0 - show_time > show_gap:
                show_time = timer()
                print(f"|{batch_size:>4}/{train_epoch:>4}   |loss {loss_avg:.4f}")

            if time0 - eval_time > eval_gap:
                '''evaluate_it'''
                eval_time = timer()
                model.eval()
                loss_sum_eva = 0
                correct = 0

                eval__time = int(eval__len / eval_size)
                eval__time = eval__time + 1 if eval__len % eval_size else eval__time
                with torch.no_grad():
                    for i in range(eval__time):
                        j = i * eval_size
                        inp = eval__imgs[j:j+eval_size]
                        lab = eval__labs[j:j+eval_size]
                        out = model(inp)

                        loss_sum_eva += criterion(
                            torch.softmax(out, dim=1), lab).item() * lab.shape[0]

                        predict = out.argmax(dim=1, keepdim=True)
                        int_lab = lab.argmax(dim=1, keepdim=True)
                        predict_bool = predict.eq(int_lab.view_as(predict))
                        correct += predict_bool.sum().item()

                loss_eva = loss_sum_eva / eval__len
                accuracy = correct / eval__len
                print(f"|{batch_size:>4}/{train_epoch:>4}   |loss {loss_avg:.4f}   "
                      f"|EvaLoss {loss_eva:.4f}   |Accu {accuracy:.4f}")

    print(f"TimeUsed:{timer() - start_time:4.0f}|loss {loss_avg:.4f}   "
          f"| EvaLoss {loss_eva:.4f}   |Accu {accuracy:.4f}")

    torch.save(model.state_dict(), save_path)
    file_size = os.path.getsize(save_path) / (2 ** 20)  # Byte --> KB --> MB
    print(f"\nSave: {mod_dir} | {file_size:.2f} MB")


def run_test(mod_dir, gpu_id, ):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # choose GPU:0
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model = Net().to(device)
    load_torch_model(mod_dir, model)

    in_put = np.zeros((1, 1, 28, 28), dtype=np.float32)
    in_put = torch.tensor(in_put, dtype=torch.float32, device=device)
    output = model(in_put)
    output = output.cpu().data.numpy()[0]
    print(np.argmax(output), output)


def main():
    args = Arguments()

    import sys
    args.gpu_id = sys.argv[0][-4]
    print("GPU ID:", args.gpu_id)
    del sys

    run_train__data_loader(args)
    run_train(args)
    # run_test(args.mod_dir, args.gpu_id)


if __name__ == '__main__':
    main()
