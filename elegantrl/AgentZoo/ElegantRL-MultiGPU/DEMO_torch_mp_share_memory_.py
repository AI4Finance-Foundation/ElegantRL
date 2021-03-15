import torch.multiprocessing as mp
import torch

"""Pytorch Multiprocessing
how-to-share-a-list-of-tensors-in-pytorch-multiprocessing
https://stackoverflow.com/a/50873015/9293137
"""


def foo(worker, t1):
    t1[worker] += (worker + 1) * 1000


if __name__ == '__main__':
    tl = [torch.rand(2), torch.rand(3)]

    for t in tl:
        t.share_memory_()

    p0 = mp.Process(target=foo, args=(0, tl))
    p1 = mp.Process(target=foo, args=(1, tl))

    print(tl)
    p0.start()
    print(tl)
    p1.start()
    print(tl)
    p0.join()
    print(tl)
    p1.join()
    print(tl)
