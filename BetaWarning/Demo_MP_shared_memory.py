import multiprocessing as mp

"""
Refer: Shared memory in multiprocessing
https://stackoverflow.com/questions/14124588/shared-memory-in-multiprocessing
https://docs.python.org/3/library/multiprocessing.html#sharing-state-between-processes
"""
import time
import numpy as np


def process1(num, ary):
    print('p1 start', num.value, ary[:])
    time.sleep(1)
    num.value = 1
    print('p1', np.array(ary).dtype)
    ary[:] = [-1 * i for i in ary]
    ary[:] = list(-np.ones(4))
    print('p1 changed')
    print('p1 print', num.value, ary[:])
    time.sleep(123)


def process2(num, ary):
    print('p2 start', num.value, ary[:])
    time.sleep(2)
    print('p2 print', num.value, ary[:])

    a = np.array(ary)
    print(a + 1)
    time.sleep(123)


if __name__ == '__main__':
    shared_num = mp.Value('d', 0.0)  # 'd': float, 'i': int
    shared_ary = mp.Array('d', range(4))
    # from multiprocessing import shared_memory # new in Python3.8
    # shared_mem = shared_memory.SharedMemory(create=True, size=4)

    process = [
        mp.Process(target=process1, args=(shared_num, shared_ary)),
        mp.Process(target=process2, args=(shared_num, shared_ary)),
    ]
    [p.start() for p in process]
    [p.join() for p in process]
