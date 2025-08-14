# IEOR4575 Project
Instructor: Professor Shipra Agrawal\
Assistants: Yunhao Tang, Abhi Gupta

## Info
The final report generated can be found under `4575_final_report_tkt2120_submission.pdf`

---------
## To Run
To run any of the training scripts on instances generated or downloaded, use
```
python run_ppo.py
```
For now, hyperparameters must be changed within the `.py` file. 

## State-Action Description

State s is an array with give components

* s[0]:  constraint matrix $A$of the current LP ($\max  -c^Tx \text{ s.t. }Ax \le  b$) . Dimension is $m \times n$. See by printing s[0].shape. Here $n$ is the (fixed) number of variables. For instances of size 60 by 60 used in the above command, $n$ will remain fixed as 60. And $m$ is the current number of constraints. Initially, $m$ is to the number of constraints in the IP instance. (For instances generated with --num-c=60, $m$ is 60 at the first step).  But $m$ will increase by one in every step of the episode as one new constraint (cut) is added on taking an action.
* s[1]: rhs $b$ for the current LP ($Ax\le b$). Dimension same as the number $m$ in matrix A.
* s[2]: coefficient vector $c$ from the LP objective ($-c^Tx$). Dimension same as the number of variables, i.e., $n$.
* s[3],  s[4]: Gomory cuts available in the current round of Gomory's cutting plane algorithm. Each cut $i$ is of the form $D_i x\le d_i$.   s[3] gives the matrix $D$ (of dimension $k \times n$) of cuts and s[4] gives the rhs $d$ (of dimension $k$). The number of cuts $k$ available in each round changes, you can find it out by printing the size of last component of state, i.e., s[4].size or s[-1].size.

## Example
You can use the following script to familiarize yourself with the cutting plane environment that we have built for you. 
```
$ python example.py
```

## Training Performance Evaluation
There are two environment settings on which your training performance will be evaluated. These can be loaded by using the following two configs (see example.py). Each mode is characterized by a set of parameters that define the cutting plane environment.

The easy setup defines the environment as follows:
```
easy_config = {
    "load_dir"        : 'instances/train_10_n60_m60',
    "idx_list"        : list(range(10)),
    "timelimit"       : 50,
    "reward_type"     : 'obj'
}
```
For your reference, the maximum total sum of rewards achievable in any given episode in the easy mode is 2.947 +- 0.5469.


The hard setup defines the environment as follows:
```
hard_config = {
    "load_dir"        : 'instances/train_100_n60_m60',
    "idx_list"        : list(range(99)),
    "timelimit"       : 50,
    "reward_type"     : 'obj'
}
```
On average, the maximum total sum of rewards achievable in any given episode in the hard mode is 2.985 +- 0.8427.

The main difference between the easy and hard modes is the number of training instances. Easy contains 10 instances while hard contains 100. Please read the ```example.py``` script would further details about what these environment parameters mean. 

## Generalization
For the first phase of the project, your task is to reach the best possible performance on the two training modes described above. We will introduce another test mode for the environment later in the semester where your agent will be tested on a cutting plane environment with unseen instances (of size 60 by 60).

## Generating New Instances

To make sure your algorithm generalizes to instances beyond those in the instances folder, you can create new environments with random IP instances and train/test on those. To generate new instances, run the following script. This will create 100 new instances with 60 constraints and 60 variables.

```
$ python generate_randomip.py --num-v 60 --num-c 60 --num-instances 100
```
The above instances will be saved in a directory named 'instances/randomip_n60_m60'. Then, we can load instances into gym env and train a cutting agent. The following code loads the 50th instance and run an episode with horizon 50:

```
python testgymenv.py --timelimit 50 --instance-idx 50 --instance-name randomip_n60_m60
```

We should see the printing of step information till the episode ends.

If you do not provide --instance-idx, then the environment will load random instance out of the 100 instances in every episode. It is sometimes easier to train on a single instance to start with, instead of a pool of instances.
