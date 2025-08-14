from pdb import set_trace as bp
from multiprocessing import Pool, Manager
from time import time
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize
import pickle as pkl

def mmse_beamformers(H, P):
    K,N = H.shape
    W = la.inv(np.eye(N)*K/P + H.T.conj()@H)@H.T.conj()
    W = W/la.norm(W,axis=0,keepdims=True)/np.sqrt(K)

    return W

def sum_rate(W,H,callback=False):
    HW = H@W
    absHW2 = np.abs(HW)**2
    S = np.diagonal(absHW2)
    I = np.sum(absHW2, axis=-1) - S
    N = 1
    
    SINR = S/(I+N)
    return np.log2(1+SINR).sum()
def print_progress(xk,):
    pass
    # N = K = 8
    # Hall = np.load('./HallN'+str(N)+'K'+str(K)+'.npy')
    # sum_rate_max = 0
    # for i in range(10):
    #     # print(Hall.shape, np.mean(Hall), np.std(Hall))
    #     W = real_to_complex(xk,N,K)
    #     # print(W)
    #     W = np.sqrt(10 ** 1) * W/la.norm(W)
    #     # print(W)
    #     s = sum_rate(W, Hall[i],True)
    #     if s > sum_rate_max:
    #         sum_rate_max = s
    # print("Current obj: {}".format(sum_rate_max))
    # assert 0
def slsqp(H, storage, SNRdb):
    K,N = H.shape


    sumrate = []
    Wall = []
    for P in 10**(SNRdb/10):
        fun = partial(sum_rate, H=H)
        consW = {'fun':partial(W_constraint, P=P, N=N, K=K), 'type':'ineq'}

        W = mmse_beamformers(H,P) 
        W0 = W * np.sqrt(P) / la.norm(W)

        x0 = complex_to_real(W0)
        
        sol = minimize(lambda z: -fun(real_to_complex(z,N,K)), x0=x0, method='CG', options={'maxiter': 50}, constraints=[consW], callback=print_progress)
        W = real_to_complex(sol.x,N,K)
        W = np.sqrt(P) * W/la.norm(W)
        sumrate.append(sum_rate(W, H))
        Wall.append(W)

    storage.append({'H':H, 'W':Wall, 'sumrate':sumrate})
    formatted_list = ["{:.2f}".format(num) for num in storage[-1]['sumrate']]
    print(len(storage), formatted_list)

def real_to_complex(z, N, K):     
    W = (z[:len(z)//2] + 1j * z[len(z)//2:]).reshape((N,K))
    return W

def complex_to_real(W):
    return np.concatenate((W.real.ravel(), W.imag.ravel()))

def W_constraint(z,P,N,K):
  W = (z[:len(z)//2] + 1j * z[len(z)//2:]).reshape((N,K))
  return P-la.norm(W)**2

if __name__ == '__main__':
    M = N = K = 128
    manager = Manager()
    SNRdb = np.array([20])
    nSamples = 10
    # load samples
    if K==4:
        with open("./test_channel/K4N4Samples120.pkl", 'rb') as f:
            Hall = pkl.load(f)#.transpose(-1,-2)
    elif K==8:
        with open("./test_channel/K8N8Samples100.pkl", 'rb') as f:
            Hall = pkl.load(f).numpy()
    else:
        
        Hall = (np.random.randn(nSamples,N,K) + 1j*np.random.randn(nSamples,N,K))/np.sqrt(2)
        np.save('./HallN'+str(N)+'K'+str(K)+'.npy', Hall)
    
    # slsqp
    # storage_slsqp = manager.list()
    storage_slsqp = []
    # with Pool() as pool:
    #     pool.map(partial(slsqp, storage=storage_slsqp, SNRdb=SNRdb), Hall)
    for i in range(nSamples):
        slsqp(Hall[i], storage_slsqp, SNRdb)
    sumrate_slsqp = np.array([s['sumrate'] for s in storage_slsqp])
    sumrate_slsqp_avg = np.mean(sumrate_slsqp, axis=0)
    print('slsqp', sumrate_slsqp_avg.tolist())

    plt.plot(SNRdb, sumrate_slsqp_avg)
    plt.legend(['SLSQP'])
    plt.grid(linestyle='--')
    plt.show()