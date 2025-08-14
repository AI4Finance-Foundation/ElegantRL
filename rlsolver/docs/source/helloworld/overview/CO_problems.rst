CO Problems
===========

We show the basic denotations of graphs. Let :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E}, W)` denote a weighted graph, where  
:math:`\mathcal{V}` is the node set, :math:`\mathcal{E}` is the edge set, :math:`|\mathcal{V}| = V`, :math:`|\mathcal{E}| = E`,  
and :math:`W : \mathcal{E} \rightarrow \mathbb{R}^+` is the edge weight function, i.e.,  
:math:`W_{u,v}` is the weight of edge :math:`(u,v) \in \mathcal{E}`. :math:`W_{u,v} > 0` if :math:`(u,v)` is an edge and 0 otherwise.  
Let :math:`\delta^+(i)` and :math:`\delta^-(i)` denote the out-arcs and in-arcs of node :math:`i`.

Integer linear programming (ILP) is a standard formulation of combinatorial optimization problems.  
It has the *canonical form*:

.. math::
   :nowrap:

   \begin{equation}
   \begin{aligned}
   \min \quad & \mathbf{c}^T \mathbf{x} \\
   \text{s.t.} \quad & A \mathbf{x} \le \mathbf{b}, \\
   & \mathbf{x} \ge 0, \\
   & \mathbf{x} \in \mathbb{Z}^n,
   \end{aligned}
   \tag{1}
   \end{equation}

where :math:`\mathbf{x}` is a vector of :math:`n` decision variables, :math:`\mathbf{c}` is a vector of :math:`n` coefficients for :math:`\mathbf{x}` in the objective function,  
:math:`A \in \mathbb{R}^{m \times n}` and :math:`\mathbf{b} \in \mathbb{R}^m` together denote :math:`m` linear constraints,  
and :math:`\mathbf{x} \in \mathbb{Z}^n` implies that we are interested in integer solutions.  
Let :math:`\mathbf{x}^*` denote the optimal solution and :math:`f^*` denote the corresponding objective value.  
Only a few problems such as portfolio optimization are quadratic programming, which will be described later.

With respect to QUBO or Ising model, we consider a 1D Ising model with a ring structure and  
an external magnetic field :math:`h_i`. There are :math:`N` nodes with :math:`(N + 1) \equiv 1 \mod N`;  
a node :math:`i` has a spin :math:`s_i \in \{+1, -1\}` (where +1 for up and −1 for down).  
Two adjacent sites :math:`i` and :math:`i + 1` have an energy  
:math:`w_{i,i+1}` or :math:`-w_{i,i+1}` depending on whether they have the same or different directions, respectively.

The whole system will evolve into the ground state with the minimum Hamiltonian:

.. math::
   :nowrap:

   \begin{equation}
   f(\mathbf{s}) = - \sum_{i=1}^N h_i s_i + \alpha \sum_{i=1}^N -w_{i,i+1} s_i s_{i+1}
   \tag{2}
   \end{equation}

where :math:`\alpha` is a weight, :math:`f_A` is defined on each node’s effect on its own,  
and :math:`f_B` is defined on each two adjacent nodes’ interactions.  
In fact, we generally use binary variables (0 or 1) to formulate the objective function,  
and :math:`s_i \in \{+1, -1\}` can be replaced by

.. math::
   :nowrap:

   \begin{equation}
   x_i = \frac{s_i + 1}{2},
   \tag{3}
   \end{equation}

where :math:`x_i \in \{0, 1\}`.


Graph Maxcut
-----------------

.. figure:: /_static/maxcut.png
   :width: 30%
   :align: center
   :alt: An example of graph maxcut

   **Figure 1: An example of graph maxcut.**

The graph maxcut problem is defined as follows.  
Given a graph :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E}, W)`,  
split :math:`\mathcal{V}` into two subsets :math:`\mathcal{V}^+` (with edge set :math:`\mathcal{E}^+`) and  
:math:`\mathcal{V}^-` (with edge set :math:`\mathcal{E}^-`),  
and the cut set is :math:`\delta = \{(i, j) \mid i \in \mathcal{V}^+, j \in \mathcal{V}^-\}`.  
The goal is to maximize the cut value:

.. math::

   \max \sum_{(i,j)\in \delta} W_{ij}
Where :math:`W_{ij}` represents the importance (or cost) of edge :math:`(i, j)`, and is usually predefined based on the graph structure.


1. ILP Formulation
~~~~~~~~~~~~~~~~~~

**Take Figure 1 for Example**

.. math::

   \max \quad \mathbf{y}_{12} + \mathbf{y}_{14} + \mathbf{y}_{23} + \mathbf{y}_{35} + \mathbf{y}_{45}

s.t.

.. math::

   \mathbf{y}_{12} \le \mathbf{x}_1 + \mathbf{x}_2 \\
   \mathbf{y}_{12} \le 2 - \mathbf{x}_1 - \mathbf{x}_2 \\
   \mathbf{y}_{12} \ge \mathbf{x}_1 - \mathbf{x}_2 \\
   \mathbf{y}_{12} \ge -\mathbf{x}_1 + \mathbf{x}_2 \\

   \mathbf{y}_{14} \le \mathbf{x}_1 + \mathbf{x}_4 \\
   \mathbf{y}_{14} \le 2 - \mathbf{x}_1 - \mathbf{x}_4 \\
   \mathbf{y}_{14} \ge \mathbf{x}_1 - \mathbf{x}_4 \\
   \mathbf{y}_{14} \ge -\mathbf{x}_1 + \mathbf{x}_4 \\

   \mathbf{y}_{23} \le \mathbf{x}_2 + \mathbf{x}_3 \\
   \mathbf{y}_{23} \le 2 - \mathbf{x}_2 - \mathbf{x}_3 \\
   \mathbf{y}_{23} \ge \mathbf{x}_2 - \mathbf{x}_3 \\
   \mathbf{y}_{23} \ge -\mathbf{x}_2 + \mathbf{x}_3 \\

   \mathbf{y}_{35} \le \mathbf{x}_3 + \mathbf{x}_5 \\
   \mathbf{y}_{35} \le 2 - \mathbf{x}_3 - \mathbf{x}_5 \\
   \mathbf{y}_{35} \ge \mathbf{x}_3 - \mathbf{x}_5 \\
   \mathbf{y}_{35} \ge -\mathbf{x}_3 + \mathbf{x}_5 \\

   \mathbf{y}_{45} \le \mathbf{x}_4 + \mathbf{x}_5 \\
   \mathbf{y}_{45} \le 2 - \mathbf{x}_4 - \mathbf{x}_5 \\
   \mathbf{y}_{45} \ge \mathbf{x}_4 - \mathbf{x}_5 \\
   \mathbf{y}_{45} \ge -\mathbf{x}_4 + \mathbf{x}_5

Each :math:`\mathbf{x}, \mathbf{y}\in \{0, 1\}`.

This ILP formulation ensures that :math:`\mathbf{y}_{ij} = 1` if and only if nodes :math:`i` and :math:`j` belong to different subsets (i.e., edge :math:`(i,j)` is cut).

**We provide general ILP formulation of graph maxcut**

.. raw:: html

    <div style="text-align: center;">
    $$\begin{array}{ll}
    \text{max} & \sum_{(i,j)} W_{ij} y_{ij} \\
    \text{s.t.} & y_{ij} \le x_i + x_j,\quad \forall i, j \in V,\ i < j \\
    & y_{ij} \le 2 - x_i - x_j,\quad \forall i, j \in V,\ i < j \\
    & y_{ij} \ge x_i - x_j,\quad \forall i, j \in V,\ i < j \\
    & y_{ij} \ge -x_i + x_j,\quad \forall i, j \in V,\ i < j
    \end{array}$$
    </div>

where :math:`x_i` is a binary variable denoting if node :math:`i` belongs to the selected subset; and :math:`y_{ij}` is 1 if nodes :math:`i` and :math:`j` are in different subsets and 0 otherwise.  
The weight :math:`W_{ij}` represents the importance (or cost) of edge :math:`(i, j)`, and is usually predefined based on the graph structure.

2. QUBO Formulation
~~~~~~~~~~~~~~~~~~~

**Take Figure 1 for example**

For an illustrative example in the left graph of **Figure 1**, the edge set is:

:math:`E = \{(1,2), (1,4), (2,3), (2,4), (3,5)\}`

and the weights are:

:math:`w_{12} = w_{14} = w_{23} = w_{24} = w_{35} = w_{45} = 1`.

The edge set of black nodes is :math:`\mathcal{E}^+ = \{(1, 4)\}`,  
and the edge set of white nodes is :math:`\mathcal{E}^- = \emptyset`.

The edges connecting the two subsets are:

:math:`\delta = \{(1,2), (2,3), (2,4), (3,5), (4,5)\}`.

The solution is :math:`x \in \{0,1\}^5`, and the Hamiltonian in Equation (3) becomes:

.. math::

   \begin{aligned}
   \min f(x) =\ 
   & -\frac{1}{2} (1 - (2x_1 - 1)(2x_2 - 1)) \\
   & -\frac{1}{2} (1 - (2x_1 - 1)(2x_4 - 1)) \\
   & -\frac{1}{2} (1 - (2x_2 - 1)(2x_3 - 1)) \\
   & -\frac{1}{2} (1 - (2x_3 - 1)(2x_5 - 1)) \\
   & -\frac{1}{2} (1 - (2x_4 - 1)(2x_5 - 1))
   \end{aligned}



**We provide general QUBO formulation of maxcut**

.. math::

   \min f(x) = -\frac{1}{2} \sum_{(i,j)\in \mathcal{E}} W_{ij} \left(1 - (2x_i - 1)(2x_j - 1) \right)

Here, :math:`\sum_{(i,j)\in \mathcal{E}} W_{ij}` is a constant.  
Let :math:`x_i = 1` if node :math:`i` belongs to :math:`V^+`, and 0 otherwise.  
The term :math:`1 - (2x_i - 1)(2x_j - 1)` equals 1 if nodes :math:`i` and :math:`j` are in different subsets, and 0 otherwise.

3. RL Methods: Two Decision Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. figure:: /_static/fig2.png
   :width: 120%
   :align: center

   **Figure 2: Two patterns for graph maxcut.**

**Pattern I**  
Initial state is empty. RL agent selects node 1 with the highest Q-value.  
Reward is 2. New state becomes :math:[1].

**Pattern II**  
Current state is :math:[2, 3], objective = 2.  
Agent adds node 1. New state is :math:[1, 3, 4].  
Objective improves to 4.

Knapsack
---------
We take **knapsack** as a second probelm.

**Knapsack problem** Given a set of items :math:`\mathcal{I}`, each item :math:`i` with an integer weight :math:`W_i` and a value :math:`\mu_i`,  
determine which items to include in the collection so that the total weight is less than or equal to a given limit :math:`U`  
and the total value is maximized.

We assume there are 3 items. Their values are :math:`[1, 1, 2]`, weights are :math:`[1, 1, 1]`, and the total weight limit is :math:`2`.

1. ILP Formulation
~~~~~~~~~~~~~~~~~~

**ILP formulation for this example**

.. math::

   \max \quad x_1 + x_2 + 2x_3 \\
   \text{s.t.} \quad x_1 + x_2 + x_3 \le 2 \\
   x_1, x_2, x_3 \in \{0, 1\}

**We provide general ILP formulation of knapsack**

.. math::

   \max \sum_{i \in \mathcal{I}} \mu_i x_i \\
   \text{s.t.} \sum_{i \in \mathcal{I}} W_i x_i \le U, \\
   x_i \in \{0, 1\}

where :math:`x_i` is a binary variable (1 if item :math:`i` is in the knapsack), :math:`\mu_i` is its value, :math:`W_i` is its weight, and :math:`U` is the weight limit.

2. QUBO Formulation
~~~~~~~~~~~~~~~~~~~

**For the same example:**


We consider an item :math:`i`, and let :math:`x_i` be a binary variable with 1 denoting it is in the knapsack and 0 otherwise.  
Let :math:`y_n` for :math:`1 \leq n \leq U` be a binary variable with 1 denoting the final weight of the knapsack is :math:`n`.

The QUBO formulation of Knapsack is:

.. math::

   \min_x \left( \left( y_1 + 2y_2 \right)^2 + \left( y_1 + 2y_2 - x_1 - x_2 - x_3 \right)^2 - \alpha (x_1 + x_2 + 2x_3) \right)

Here, :math:`y_1, y_2` are binary variables indicating whether total weight equals 1 or 2 respectively. :math:`\alpha \in (0,1)` is a penalty parameter.

**We provide general QUBO formulation of knapsack**

.. math::

   \min_x f = \left( \sum_{n=1}^{U} y_n \right)^2 + \left( \sum_{n=1}^{U} n y_n - \sum_{i \in \mathcal{I}} W_i x_i \right)^2 - \alpha \sum_{i \in \mathcal{I}} \mu_i x_i

where :math:`y_n \in \{0,1\}` denotes whether the final weight equals :math:`n`, and :math:`x_i` is whether item :math:`i` is in the knapsack.


3. Two Patterns of RL Methds
~~~~~~~~~~~~~~~~~~

For the same 3-item knapsack problem

**Pattern I**  
The initial state is empty. Then we select item 2 and add it to the state,  
i.e., the new state is :math:[2]. The reward is 1.

**Pattern II**  
The current state is :math:[0, 1, 0], and the objective value is 1.  
The new state is :math:[0, 1, 1], and the objective value is 3.


