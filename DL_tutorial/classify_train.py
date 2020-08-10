import os
import time
import numpy as np
import numpy.random as rd

import torch
from yonv_utils import load_data
from yonv_utils import load_torch_model
from yonv_utils import whether_remove_history
# from classify_netwrok import Res1dNet as Net
# from classify_netwrok import Conv2dNet as Net
from classify_netwrok import SE2dNet as Net

"""
Github: Yonv1943 

net         para(MB time(s  accuracy(train_epoch == 3 or 6

Res1dNet    0.49    36      >0.88       61      >0.89
Conv2dNet   0.64    31      >0.92       50      >0.93
SE2dNet     0.67    32      >0.92       64      >0.93
"""


class Arguments:
    train_epochs = [max(int(6 * 0.6065 ** i), 1) for i in range(6)]
    batch_sizes = [int(128 * 1.6487 ** i) for i in range(6)]

    show_gap = 2 ** 1
    eval_gap = 2 ** 3

    def __init__(self, gpu_id=-1, mod_dir=''):
        import sys
        self.mod_dir = mod_dir if mod_dir != '' else sys.argv[0].split('/')[-1][:-3]
        self.gpu_id = gpu_id if gpu_id != -1 else self.mod_dir[-1]
        self.gpu_ram = 0.9  # 0.0 ~ 1.0
        del sys


def train_it(model, train_loader, device, optimizer, criterion):
    loss_sum = 0
    model.train()

    batch_size = train_loader.batch_size
    for image, target in train_loader:
        k = rd.randint(2, batch_size)
        image[:k] = image[:k].flip(3)

        image = image.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    loss_avg = loss_sum / len(train_loader)
    return loss_avg


def eval_it(model, eval_loader, device, criterion):
    model.eval()
    loss_sum = 0
    correct = 0
    with torch.no_grad():
        for image, target in eval_loader:
            image = image.to(device)
            target = target.to(device)
            output = model(image)

            loss_sum += criterion(output, target, reduction='mean').item()

            predict = output.argmax(dim=1, keepdim=True)
            predict_bool = predict.eq(target.view_as(predict))
            correct += predict_bool.float().mean().item()

    eval_loader_len = len(eval_loader)
    loss_avg = loss_sum / eval_loader_len
    accuracy = correct / eval_loader_len
    print(end=' |EVAL: Loss: %.4f |Accu: %.4f' % (loss_avg, accuracy))


def run_train(mod_dir, gpu_id,
              train_epochs, batch_sizes,
              show_gap, eval_gap):
    whether_remove_history(mod_dir, remove=True)

    '''init env'''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # choose GPU:0
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    np.random.seed(1943 + int(time.time()))
    torch.manual_seed(1943 + rd.randint(0, int(time.time())))

    '''load data'''
    import torch.utils.data as data
    train_images, train_labels, eval_images, eval_labels = load_data()

    train_sets = data.TensorDataset(train_images, train_labels)
    eval_sets = data.TensorDataset(eval_images, eval_labels)

    eval_loader = data.DataLoader(
        eval_sets, batch_size=batch_sizes[0], shuffle=False, drop_last=False, )
    del eval_sets

    '''train model'''
    from torch.nn import functional
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = functional.nll_loss

    save_path = load_torch_model(mod_dir, model)

    print("Train Loop:")
    from time import time as timer
    start_time = timer()
    eval_time = show_time = 0
    loss_avg = None
    for train_epoch, batch_size in zip(train_epochs, batch_sizes):
        print(end="\n||%d/%d\t\t" % (batch_size, train_epoch))
        for epoch in range(train_epoch):
            train_loader = data.DataLoader(
                train_sets, batch_size=batch_size, shuffle=True, drop_last=True, )
            loss_avg = train_it(model, train_loader, device, optimizer, criterion)

            time0 = timer()
            if time0 - show_time > show_gap:
                show_time = timer()
                print(end='\n  loss: %.4f' % (loss_avg,))

            if time0 - eval_time > eval_gap:
                eval_time = timer()
                eval_it(model, eval_loader, device, criterion)

    print('\n\nTime Used: %i' % (timer() - start_time))
    print(end='\n  loss: %.4f' % (loss_avg,))
    eval_it(model, eval_loader, device, criterion)

    torch.save(model.state_dict(), save_path)
    file_size = os.path.getsize(save_path) / (2 ** 20)  # Byte --> KB --> MB
    print("\nSave: %s | %.2f MB" % (mod_dir, file_size))


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


if __name__ == '__main__':
    args = Arguments(gpu_id=3)
    print('  GPUid: %s' % args.gpu_id)

    run_train(args.mod_dir, args.gpu_id,
              args.train_epochs, args.batch_sizes,
              args.show_gap, args.eval_gap)
    # run_test(args.mod_dir, args.gpu_id)
