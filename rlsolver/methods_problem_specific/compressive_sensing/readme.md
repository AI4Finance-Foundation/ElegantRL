# Compressive sensing using generative models
 
 [1] Wu, Yan, Mihaela Rosca, and Timothy Lillicrap. "Deep compressed sensing." International Conference on Machine Learning, 2019.
 
 First case, linear measurment process: $\boldsymbol{y} = \boldsymbol{F} \boldsymbol{x}$, where the true signal $\boldsymbol{x} \in \mathbb{R}^n$, $\boldsymbol{F} \in \mathbb{R}^{m \times n}$, and $\boldsymbol{y} \in \mathbb{R}^m $, $m \ll n$.

## Recovery error $\lVert x-\hat{x}\rVert_2$ for MNIST

A pretrained model $G_\theta$: $G$ is a neural network with parameters $\theta$.

- Ours: 4.78
- DCS: 3.4

Ours: Formula (7) is trained as a deep neural network.

## Recovery on the MNIST dataset

 $\boldsymbol{F}_\phi$: $\boldsymbol{F}$ is reparameterized as a deep neural network with parameter $\phi$.

|Method|LOSS|Origin image| 1 steps|3 steps | 5 steps|
|-------| ----|------- | -----|------ |-----|
|$\boldsymbol{F}_\phi$ (L) + grad|4.78|![alt_text](fig/origin.png)|![alt_text](fig/reconstruction_0.png)|![alt_text](fig/reconstruction_3.png)|![alt_text](fig/reconstruction_5.png)|
|$\boldsymbol{F}_\phi$ (L) + NN|10.20|![alt_text](fig/origin.png)|![alt_text](fig/reconstruction_0_nn.png)|![alt_text](fig/reconstruction_3_nn.png)|![alt_text](fig/reconstruction_5_nn.png)|
|Fix $\boldsymbol{F}$ + grad steps          (m = 100) |6.97|![alt_text](fig/origin.png)|![alt_text](fig/reconstruction_0_4_last.png)|![alt_text](fig/reconstruction_3_4_last.png)|![alt_text](fig/reconstruction_5_4_last.png)|
|Fix $\boldsymbol{F}$ + grad steps          (m = 300)|4.50|![alt_text](fig/origin.png)|![alt_text](fig/reconstruction_0_3_last.png)|![alt_text](fig/reconstruction_3_3_last.png)|![alt_text](fig/reconstruction_5_3_last.png)|

|${\overline{X}}$|$\overline{G_\theta (z_0)}$|
|-----|-----|
|<img src="./fig/origin_average.png"  width="50%" height="50%">|<img src="./fig/recon_average.png"  width="50%" height="50%">|

## Recovery on the synthetic sparse signal
<!-- ### DCS
|Method|Number of iterations|Origin|Recovery|
|---|----|----|----|
|LASSO|10|![alt_text](./fig/origin_signal_11.png)|![alt_text](./fig/recovery_signal_lasso.png)|
|$G_\theta(z)$|10|![alt_text](./fig/origin_signal_11.png)|![alt_text](./fig/recovery_signal_11.png)|
 -->
- The true signal $\boldsymbol{x} \in \mathbb{R}^n$  has a sparse representation $\boldsymbol{z}\in \mathbb{R}^n$ in the representation domain $\boldsymbol{\Phi} \in \mathbb{R}^{n \times n}$, where $\lVert \boldsymbol{z} \rVert_0=k$.
### Synthetic Signal
- Latent representation $\boldsymbol{z}$  $\in \mathbb{B}^{n}$, where $\mathbb{B}$ =  $\lbrace -1,0, 1\rbrace$ and $\lVert \boldsymbol{z}\rVert_0 = k$.
- Representation domain (Basis) $\boldsymbol{\Phi} \in \mathbb{R}^{n\times n}$.
- Synthetic signal $\boldsymbol{x} = \boldsymbol{\Phi} \boldsymbol{z}$.

    $n=100, k=10$
    | $\Phi$|$\boldsymbol{z}$|$\boldsymbol{x}$|
    |---|----|----|
    |Identity|fig|fig|
    |DCT|fig|fig|


### Generator $G_\theta(\boldsymbol{z})$
- The latent dimension of the generator model input, $\boldsymbol{z}$, is larger than $k$ and much less than $n$.
- Training samples: $\lbrace (\boldsymbol{z},\boldsymbol{x})\rbrace$
- Loss function:  $MSE(G_\theta(\boldsymbol{z}), \boldsymbol{x})$
- Error: $\frac{\lVert\boldsymbol{\Phi}\boldsymbol{z}-G_\theta(\boldsymbol{z})\rVert_2}{\lVert\boldsymbol{\Phi}\boldsymbol{z}\rVert_2}\times 100$%


||Lasso|DCS|Ours|
|-|--|--|--|
|Intialization|$f(\boldsymbol{F})$|$G_\theta(\boldsymbol{z})$|$G_\theta(\boldsymbol{z})$|
|Sparse Structure|$\min_{\boldsymbol{z}}\lVert \boldsymbol{z}\rVert_1$|$latentDim << N$ |$latentDim << N$|
|Iterative Method| Gradient based|Gradient based, Eqn. $(7)$| Forward propagation|
|\#Iterations (MNIST)|$10\sim30$|$5$|$\textcolor{blue}{<5}$|
|\#Iterations (Synthetic)|$10\sim30$|$\textcolor{blue}{Undoable}$|$\textcolor{blue}{<5}$|

#### Verify whether $G_\theta(z)$ approximates a 
sparse structure.
- Verify whether $G_\theta(\boldsymbol{z})$ is sparse in the representation domain $\boldsymbol{\Phi}$, namely $\lVert\boldsymbol{\Phi}^{-1}G_\theta(\boldsymbol{z})\rVert_0 \approx k$.



### Lasso (CS)
- $\widehat{\boldsymbol{z}} = Lasso(\boldsymbol{y}, \boldsymbol{F} \boldsymbol{ \Phi })$
- $\widehat{\boldsymbol{z}}_0 \xrightarrow[\sim\text{30 iterations}]{Lasso} \widehat{\boldsymbol{z}}$
- Error: $\frac{\lVert \boldsymbol{x} - \widehat{\boldsymbol{x}} \lVert_2}{\lVert \boldsymbol{x} \rVert_2} \times 100$%, where $\widehat{\boldsymbol{x}} = \boldsymbol{ \Phi } \widehat{\boldsymbol{z}} $.

 
#### $m=50 $ 

|#iterations| $5$ | $10$ |  $15$|
|------|------|------|-----|
|$n=100$|86.86%|61.76%|0.000439%|


### DCS
- Given train $G_\theta$, using Eqn. (7) in [1] to recover $\boldsymbol{\widehat{z}}$ and $\boldsymbol{\widehat{x}}$.
- $\widehat{\boldsymbol{z}}_0 \xrightarrow[\text{hundreds of iterations}]{Eqn. (7)} \widehat{\boldsymbol{z}}$
- $\widehat{\boldsymbol{x}} = \boldsymbol{ G }_\theta ( \widehat{\boldsymbol{z}} )$.

#### $m=50 $ 

|#iterations| $100$ | $200$ |  $300$|
|------|------|------|-----|
|$n=100$|75.14%|74.68%||


### Ours
- Train a neural network (NN) to replace Eqn. (7) in [1].
- $\widehat{\boldsymbol{z}}_0 \xrightarrow[\text{\\# iterations < 10}]{NN} \widehat{\boldsymbol{z}}$
- $\widehat{\boldsymbol{x}} = \boldsymbol{ G }_\theta ( \widehat{\boldsymbol{z}} )$.

#### $m=50 $ 

|#iterations| $1$ | $3$ | $5$|
|------|------|------|-----|
|$n=100$||||
