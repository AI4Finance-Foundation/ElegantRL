现在TNCO出最好结果的代码，是
https://github.com/AI4Finance-Foundation/ElegantRL_Solver/tree/main/rlsolver/rlsolver_learn2opt/tensor_train

也就是当前文件夹内的：
- `H2O_MISO.py` Learn to optimize 的代码，可以被其他代码调用，搭配有MISO 的例子
- `TNCO_env.py` TNCO 任务的环境，里面的 `get_objective` 函数可以被调用。里面还储存了我们搜索出来的解，且有代码验证这些解的得分。
- `TNCO_H2O.py` 调用`H2O_MISO.py` 里 Learn to optimize 的代码，以及`TNCO_env.py`里 TNCO 任务的环境，搜索这个问题的解`theta`

在GPU编号为 0 的设备上运行这个代码：
`python3 TNCO_H2O.py 0`

