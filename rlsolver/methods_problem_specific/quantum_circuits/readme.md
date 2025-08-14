# Classical simulation of quantum circuits using tensor networks


## Experimental Results

The quantum circuits have qubits n=53 and cycles (depth) m=12, 14, 16, 18, 20, respectively, and the datasets can be downloaded from https://datadryad.org/stash/dataset/doi:10.5061/dryad.k6t1rj8                                                                                                                                          
The file and the transformed txt file are located in sycamore_circuits/sycamore.

**Our results**

|Sycamore_Circuit|m=12|m=14|m=16|m=18|m=20|
|-------|------- | -----|------ |------ |------ |
|Results|OE_greedy: 17.494<br>CTG_Greedy: 16.764<br>CTG_Kahypar: 13.107<br>**RL: 14.780**|OE_greedy: 19.378<br>CTG_Greedy: 18.981<br>CTG_Kahypar: 13.851<br>**RL: 15.232**|OE_greedy: 25.588<br>CTG_Greedy: 22.850<br>CTG_Kahypar: 16.711<br>**RL: 18.840**|OE_greedy: 26.492<br>CTG_Greedy: 23.269<br>CTG_Kahypar: 17.383<br>**RL: 18.858**|OE_greedy: 26.982<br>CTG_Greedy: 25.322<br>CTG_Kahypar: 18.525<br>**RL: 18.816**|

- **OE_greedy**: Daniel, G., Gray, J., et al. (2018). Opt\_einsum-a python package for optimizing contraction order for einsum-like expressions. Journal of Open Source Software, 3(26):753
https://github.com/dgasmith/opt_einsum

- **CTG_Greedy、CTG_Kahypar**: Gray, J. and Kourtis, S. (2021). Hyper-optimized tensor network contraction. Quantum, 5:410.
https://github.com/jcmgray/cotengra

Results of **ICML-Optimizing tensor network contraction using reinforcement learning**

|Sycamore_Circuit | m=10 | m=12|m=14|m=16 (Not-Giving)| m=18 (Not-Giving) | m=20 |
|-------| ----|------- | -----|------ |------ |------ |
|Results|OE_greedy: 14.756<br>CTG_Greedy: 10.577<br>CTG_Kahypar: 10.304<br>RL_TNCO: 10.736|OE_greedy: 20.471<br>CTG_Greedy: 14.009<br>CTG_Kahypar: 13.639<br>RL_TNCO: 12.869|OE_greedy: 18.182<br>CTG_Greedy: 15.283<br>CTG_Kahypar: 14.704<br>RL_TNCO: 14.420|OE_greedy: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL_TNCO: |OE_greedy: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL_TNCO: |OE_greedy: 31.310<br>CTG_Greedy: 18.934<br>CTG_Kahypar: 18.765<br>RL_TNCO: 18.544|

- **TNCO**: Meirom E, Maron H, Mannor S, et al. Optimizing tensor network contraction using reinforcement learning. International Conference on Machine Learning. PMLR, 2022: 15278-15292.

![image](https://user-images.githubusercontent.com/75991833/227595309-a341713d-0247-4f3b-a12b-d94ac74af351.png)


|TT|N=100|N=200|N=400|N=600|N=800|N=1000|N=1500|N=2000|
|------- | ----|------- |----|------- | -----|------ | -----|------ |
|Results|OE_greedy: 30.626<br>CTG_Greedy: 30.404<br>CTG_Kahypar: 30.410<br>**RL: 30.404**|OE_greedy: 60.729<br>CTG_Greedy: 60.507<br>CTG_Kahypar: 60.510<br>**RL: 60.507**|OE_greedy: 120.935<br>CTG_Greedy: 120.713<br>CTG_Kahypar: 120.713<br>**RL: 120.713**|OE_greedy:  181.141<br>CTG_Greedy: 180.919<br>CTG_Kahypar: 180.919<br>**RL: 180.919**|OE_greedy: 241.347<br>CTG_Greedy: 241.125<br>CTG_Kahypar: 241.129<br>**RL: 241.125**|OE_greedy: 301.553<br>CTG_Greedy: 301.331<br>CTG_Kahypar: 301.331<br>**RL: 301.331**|OE_greedy: N<br>CTG_Greedy: N<br>CTG_Kahypar: 451.849<br>**RL: 451.846**|OE_greedy: N<br>CTG_Greedy: N<br>CTG_Kahypar: 602.361<br>**RL: 602.361**|

|TR|N=100|N=200|N=400|N=600|N=800|N=1000|N=1500|N=2000|
|-------| -----|------| ----|------- | -----|------ | -----|------ |
|Results|OE_greedy: 30.927<br>CTG_Greedy: 30.705<br>CTG_Kahypar: 30.709<br>**RL: 30.705**|OE_greedy: 61.030<br>CTG_Greedy: 60.808<br>CTG_Kahypar: 60.809<br>**RL: 60.808**|OE_greedy: 121.236<br>CTG_Greedy: 121.014<br>CTG_Kahypar: 121.019<br>**RL: 121.014**|OE_greedy: 181.442<br>CTG_Greedy: 181.220<br>CTG_Kahypar: 181.220<br>**RL: 181.220**|OE_greedy: 241.648<br>CTG_Greedy: 241.426<br>CTG_Kahypar: 241.429<br>**RL: 241.426**|OE_greedy: 301.854<br>CTG_Greedy: 301.632<br>CTG_Kahypar: 301.629<br>**RL: 301.632**|OE_greedy: <br>CTG_Greedy: <br>CTG_Kahypar: 452.147<br>**RL: 452.147**|OE_greedy: <br>CTG_Greedy: <br>CTG_Kahypar: 602.662<br>**RL: 602.662**|



|TTN|Height=3|Height=4|Height=5|Height=6|Height=7|Height=8|Height=9|Height=10|
|-------| ----|------- | -----|------ |------ |------ |------ |------ |
|Results|OE_greedy: 3.003<br>CTG_Greedy: 2.857<br>CTG_Kahypar: 2.859<br>RL: **2.857**|OE_greedy: 5.175<br>CTG_Greedy: 5.126<br>CTG_Kahypar: 5.129<br>RL: **5.126**|OE_greedy: 9.937<br>CTG_Greedy: 9.934<br>CTG_Kahypar: 9.939<br>RL: **9.934**|OE_greedy: 19.567<br>CTG_Greedy: 19.567<br>CTG_Kahypar: 19.569<br>RL: **19.567**|OE_greedy: 38.833<br>CTG_Greedy: 38.833<br>CTG_Kahypar: 38.833<br>RL: **38.833**|OE_greedy: 77.365<br>CTG_Greedy: 77.365<br>CTG_Kahypar: 77.369<br>RL: **77.365**|OE_greedy: 154.428<br>CTG_Greedy: 154.428<br>CTG_Kahypar: 154.429<br>RL: **154.428**|OE_greedy: N<br>CTG_Greedy: N<br>CTG_Kahypar: 308.559<br>RL: **308.556**|



|MERA|Height=3|Height=4|Height=5|Height=6|Height=7|Height=8|Height=9|Height=10|
|-------| ----|------- | -----|------ |------ |------ |------ |------ |
|Results|OE_greedy: 3.595<br>CTG_Greedy: 3.609<br>CTG_Kahypar: 3.600<br>RL: **XXX**|OE_greedy: 5.793<br>CTG_Greedy: 5.393<br>CTG_Kahypar: 5.390<br>RL: **XXX**|OE_greedy: 11.446<br>CTG_Greedy: 10.111<br>CTG_Kahypar: 10.110<br>RL: **XXX**|OE_greedy: 21.079<br>CTG_Greedy: 19.743<br>CTG_Kahypar: 19.740<br>RL: **XXX**|OE_greedy: 39.009<br>CTG_Greedy: 39.009<br>CTG_Kahypar: 39.010<br>RL: **XXX**|OE_greedy: 77.541<br>CTG_Greedy: 77.541<br>CTG_Kahypar: 77.540<br>RL: **XXX**|OE_greedy: 154.604<br>CTG_Greedy: 154.604<br>CTG_Kahypar: 154.600<br>RL: **XXX**|OE_greedy: N<br>CTG_Greedy: N<br>CTG_Kahypar: 308.730<br>RL: **XXX**|

|PEPS|N=36|N=64|N=100|N=144|N=196|N=256|
|-------| ----|------- | -----|------ |------ |------ |
|Results|OE_greedy: 12.996<br>CTG_Greedy: 12.944<br>CTG_Kahypar: 12.590<br>RL: **XXX**|OE_greedy: 21.983<br>CTG_Greedy: 21.975<br>CTG_Kahypar: 21.050<br>RL: **XXX**|OE_greedy: 34.317<br>CTG_Greedy: 33.715<br>CTG_Kahypar: 31.890<br>RL: **XXX**|OE_greedy: 48.165<br>CTG_Greedy: 47.262<br>CTG_Kahypar: 45.140<br>RL: **XXX**|OE_greedy: 64.420<br>CTG_Greedy: 64.420<br>CTG_Kahypar: 60.790<br>RL: **XXX**|OE_greedy: 83.084<br>CTG_Greedy: 82.783<br>CTG_Kahypar: 78.850<br>RL: **XXX**|


