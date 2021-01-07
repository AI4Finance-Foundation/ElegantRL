import multiprocessing as mp


if __name__ == '__main__':
    pipe1, pipe2 = mp.Pipe(duplex=True)

    pipe1.send_bytes(b'1')

    print(pipe2.poll(1), pipe1.poll())

    rec = pipe2.recv_bytes()
    print(pipe2.poll(1))
    print(rec, str(rec))
