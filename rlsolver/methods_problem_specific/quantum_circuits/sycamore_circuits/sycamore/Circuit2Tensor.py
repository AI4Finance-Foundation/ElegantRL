from os.path import join, dirname, exists, abspath

import sys

import time

start = time.perf_counter()
log = open('log.txt', mode='a',encoding='utf-8')
sys.path.append(abspath(dirname(__file__)).strip('examples'))

from bighead import (
    get_tensors,
)
from bighead.args import args

'''
Commandline example:
python examples/circuit_simulation.py -n 53 -m 12 -simplify -sc 36 -tc 41 -cuda 0 # for n53m12ABCD circuit
'''

PACKAGEDIR = abspath(join(dirname(__file__), '../bighead/'))

def sycamore_circuit_simulation():
        tensors, labels, final_qubits_representation = get_tensors(args.n, args.m, seq=args.seq, simplify=args.simplify)
        # print("Tensors is: ",tensors, file=log)
        print(labels, file=log)
        # print("Labels is ", labels, file=log)

if __name__ == '__main__':
    sycamore_circuit_simulation()

end = time.perf_counter()
log.close() # 关闭文件
print('Running time: %s Seconds' % (end - start))
