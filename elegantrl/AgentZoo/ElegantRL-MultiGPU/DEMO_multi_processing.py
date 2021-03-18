import time
import numpy as np
import multiprocessing as mp

"""An Tutorial of multi-processing (a Python built-in library)
"""


def func_pipe1(conn, p_id):
    print(p_id)

    time.sleep(0.1)
    conn.send(f'{p_id}_send1')
    print(p_id, 'send1')

    time.sleep(0.1)
    conn.send(f'{p_id}_send2')
    print(p_id, 'send2')

    time.sleep(0.1)
    rec = conn.recv()
    print(p_id, 'recv', rec)

    time.sleep(0.1)
    rec = conn.recv()
    print(p_id, 'recv', rec)


def func_pipe2(conn, p_id):
    print(p_id)

    time.sleep(0.1)
    conn.send(p_id)
    print(p_id, 'send')
    time.sleep(0.1)
    rec = conn.recv()
    print(p_id, 'recv', rec)


def func1(i):
    time.sleep(1)
    print(f'args {i}')


def func2(args):  # multiple parameters (arguments)
    # x, y = args
    x = args[0]  # write in this way, easier to locate errors
    y = args[1]  # write in this way, easier to locate errors

    time.sleep(1)  # pretend it is a time-consuming operation
    return x - y


def run__pool():  # main process
    from multiprocessing import Pool

    cpu_worker_num = 3
    process_args = [(1, 1), (9, 9), (4, 4), (3, 3), ]

    print(f'| inputs:  {process_args}')
    start_time = time.time()
    with Pool(cpu_worker_num) as p:
        outputs = p.map(func2, process_args)
    print(f'| outputs: {outputs}    TimeUsed: {time.time() - start_time:.1f}    \n')

    '''Another way (I don't recommend)
    Using 'functions.partial'. See https://stackoverflow.com/a/25553970/9293137
    from functools import partial
    # from functools import partial
    # pool.map(partial(f, a, b), iterable)
    '''


def run__process():  # mp: multiprocessing
    from multiprocessing import Process
    process = [Process(target=func1, args=(1,)),
               Process(target=func1, args=(2,)), ]
    [p.start() for p in process]
    [p.join() for p in process]


def run__pipe():
    from multiprocessing import Process, Pipe

    conn1, conn2 = Pipe()

    process = [Process(target=func_pipe1, args=(conn1, 'I1')),
               Process(target=func_pipe2, args=(conn2, 'I2')),
               Process(target=func_pipe2, args=(conn2, 'I3')), ]

    [p.start() for p in process]
    print('| Main', 'send')
    conn1.send(None)
    print('| Main', conn2.recv())
    [p.join() for p in process]


def run__queue():
    from multiprocessing import Process, Queue

    queue = Queue(maxsize=4)  # the following attribute can call in parent(main) or child process, just like 'Pipe'
    queue.put(True)
    queue.put([0, None, object])  # you can put deepcopy thing
    queue.qsize()  # the length of queue
    print(queue.get())  # First In First Out
    print(queue.get())  # First In First Out
    queue.qsize()  # the length of queue

    process = [Process(target=func1, args=(queue,)),
               Process(target=func1, args=(queue,)), ]
    [p.start() for p in process]
    [p.join() for p in process]


if __name__ == '__main__':  # it is necessary to write main process in "if __name__ == '__main__'"
    # run__process()
    # run__pool()
    run__pipe()
