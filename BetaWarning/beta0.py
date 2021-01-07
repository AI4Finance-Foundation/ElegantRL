import time

import numpy as np

"""An Tutorial of multi-processing (a Python built-in library)
"""


def function_pipe1(conn):
    p_id = 1
    print(p_id, 0)
    time.sleep(1)

    conn.send(np.ones(1))
    print(p_id, 'send1')
    ary = conn.recv()
    print(p_id, 'recv1', ary.shape)
    conn.send(np.ones(1))
    print(p_id, 'send2')

    time.sleep(3)


def function_pipe2(conn):
    p_id = '\t\t2'
    print(p_id, 0)
    time.sleep(1)

    conn.send(np.ones(2))
    print(p_id, 'send1')

    ary = conn.recv()
    print(p_id, 'recv1', ary.shape)
    ary = conn.recv()
    print(p_id, 'recv2', ary.shape)

    time.sleep(3)


def func1(x):  # single parameter (argument)
    time.sleep(1)  # pretend it is a time-consuming operation
    return x - 1


def func2(x, y):  # multiple parameters (arguments)
    time.sleep(1)  # pretend it is a time-consuming operation
    return x - y


def func3(args):  # multiple parameters (arguments)
    # x, y = args
    x = args[0]  # write in this way, easier to locate errors
    y = args[1]  # write in this way, easier to locate errors

    time.sleep(1)  # pretend it is a time-consuming operation
    return x - y


def demo1__pool():  # main process
    from multiprocessing import Pool

    # Source: https://docs.python.org/3/library/multiprocessing.html
    # Modify: Github Yonv1943
    cpu_worker_num = 3

    print("==== input is single parameter (argument)")
    process_args = [1, 9, 4, 3]

    print(f'| mp.Pool    inputs:  {process_args}')
    start_time = time.time()
    with Pool(cpu_worker_num) as p:
        outputs = p.map(func1, process_args)
    print(f'| mp.Pool    outputs: {outputs}    TimeUsed: {time.time() - start_time:.1f}    \n')

    print(f'| single     inputs:  {process_args}')
    start_time = time.time()
    outputs = [func1(x) for x in process_args]
    print(f'| single     outputs: {outputs}    TimeUsed: {time.time() - start_time:.1f}    \n')

    print("==== input is multiple parameters (arguments)")
    process_args = [(1, 1), (9, 9), (4, 4), (3, 3), ]

    print("| Way1: Change func2 into func3: x, y = args\n"
          "|       Now multiple arguments become single arguments.")

    print(f'|       inputs:  {process_args}')
    start_time = time.time()
    with Pool(cpu_worker_num) as p:
        outputs = p.map(func3, process_args)
    print(f'|       outputs: {outputs}    TimeUsed: {time.time() - start_time:.1f}    \n')

    print("| Way2: Using 'functions.partial' and not need to change func2\n"
          "|       See https://stackoverflow.com/a/25553970/9293137")


if __name__ == '__main__':  # it is necessary to write main process in "if __name__ == '__main__'"
    # demo1__pool()
    # conn1, conn2 = Pipe()
    #
    # process = [
    #     Process(target=function_pipe1, args=(conn1,)),
    #     Process(target=function_pipe2, args=(conn2,)),
    # ]
    #
    # [p.start() for p in process]
    # [p.join() for p in process]

    res = np.array((680/101, 820/177)).round(3)
    print(res)
    res = np.array((512/72, 1230/200, 1020/184,
                    512/73, 614/88,
                    614/84, 512/65, 614/87, 512/75)).round(3)
    print(res)
    res = np.array((207/475, 121/233, 156/344, 211/601)).round(3)
    print(res)

