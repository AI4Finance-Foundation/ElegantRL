import pickle
import time

"""
Source: Yonv1943 2019-05-04
https://github.com/Yonv1943/Python
https://zhuanlan.zhihu.com/p/64534116
Python send and receive objects through Sockets - Sudheesh Singanamalla
import socket, pickle
https://stackoverflow.com/a/47396267/9293137
Pickle EOFError: Ran out of input when recv from a socket - Antti Haapala
from multiprocessing.connection import Client
https://stackoverflow.com/a/24727097/9293137
"""


def run_client(host, port):
    data = ['any', 'object']  # the Python object you wanna send
    # import numpy as np
    # data = np.zeros((1234, 1234, 3), np.uint8)  # Pickle EOFError

    from multiprocessing.connection import Client
    client = Client((host, port))

    while True:
        data_string = pickle.dumps(data)
        client.send(data_string)

        print('Send', type(data))
        time.sleep(0.5)


def run_server(host, port):
    from multiprocessing.connection import Listener
    server_sock = Listener((host, port))
    print('Server Listening')

    conn = server_sock.accept()
    print('Server Accept')
    while True:
        data_bytes = conn.recv()
        data = pickle.loads(data_bytes)
        print('Received:', type(data))

        data_bytes = pickle.dumps(data)
        conn.send(data_bytes)


if __name__ == '__main__':
    server_host = '172.17.0.8'  # host = 'localhost'
    server_port = 22  # if [Address already in use], use another port

    import socket


    def get_ip_address(remote_server="8.8.8.8"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((remote_server, 80))
        return s.getsockname()[0]

    if get_ip_address() == server_host:
        run_server(server_host, server_port)  # first, run this function only in server
    else:
        run_client(server_host, server_port)  # then, run this function only in client