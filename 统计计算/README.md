## 随机数的生成
### Inverse Transforms

$$
\begin{align}
P(F^{-1}_X(U)\leq x)=F_x(x)
\end{align}
$$

### Acceptance-Rejection
对于密度函数$f(x)$,$g(x)$为一密度函数，与$f(x)$有相同支撑,且

$$
\frac{f(x)}{g(x)}\leq c
$$

步骤：
1) 从分布$g(x)$产生一个随机数$y$，从均匀分布产生一个随机数$U$
2) 若$U<\frac{f(y)}{cg(y)}$接受y作为f(x)的一个抽样，否则拒绝  
证明：

$$
\begin{aligned}
P(accept|Y)=P(U<\frac{f(Y)}{g(Y)})=\frac{f(y)}{cg(y)}\\
\sum_yP(accept|y)P(Y=y)=\sum_y\frac{f(y)}{cg(y)}g(y)=\frac{1}{c}\\
P(k|accepted)=\frac{P(accepted|k)g(k)}{P(accepted)}=\frac{[f(k)/(cg(k))]g(k)}{1/c}=f(k)
\end{aligned}
$$

## Bootstrap和Jackknife
## Markov Chain蒙特卡洛方法
### MH算法
### Gibbs Sampler
## MM算法
A function $g(θ|θ(k))$ is said to majorize the function $f(θ)$ at $θ(k)$ provided

$$
\begin{align}
f(\theta)\leq g(\theta|\theta_{(k)}) for\ all\ \theta \\
f(\theta_{(k)})=g(\theta_{(k)}|\theta_{(k)})
\end{align}
$$

$\theta_{(k+1)}=\underset{\theta}{argmin}g(\theta|\theta_{(k)})$
### 用于构造函数的不等式
1) Property of convex function(凸函数的性质)

   $$
   K(\lambda\theta_1+(1-\lambda)\theta_2)\leq \lambda K(\theta_1)+(1-\lambda)K(\theta_2)
   $$

2) Jensen inequality  
   for a convex function $K(x)$

   $$
   K[E(x)]\leq E[K(x)]
   $$

   specially $-logEX\leq E(-logX)$
3) Arith
4) Supporting hyperplanes 支持超平面(一维即切线)  
   suppose K() is convex and differentiable  
   $K(\theta)\leq K(\theta_{(k)})+[\nabla_K(\theta^{k})]^{T}(\theta-\theta^{(k)})$
5) Cauchy-Schwartz 
6) Quadratic upper bound  
   $K(\theta)$二阶可导且有界，我们可以找到一个矩阵M，使得$M-\nabla^2(\theta)$半正定  
牛顿算法每一次迭代都需要计算Hessen矩阵的逆，MM算法只用算一次，但MM比牛顿算法慢