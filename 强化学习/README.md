# DP
流程：
初始策略
    循环
    
        策略评估
        策略改进
最终策略
## 策略迭代
### 策略评估
策略评估：对于任意一个策略$\pi$,如何计算其状态价值函数$v_\pi$。

$$
\begin{align}
v_{\pi}(s) =& E_{\pi}[G_t|S_t=s] \\
=& E_{\pi}[R_{t+1}+G_{t+1}|S_t=s] \\
=& E_{\pi}[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s] \\
=& \sum_{a}\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_k(s')]
\end{align}
$$
只要$\gamma<1$或任何状态在策略$\pi$下都能保证最后终止，那么$v_\pi$唯一存在。
但是上式计算困难，使用迭代法求解
$$
\begin{align}
v_{k+1}(s)
=& \sum_{s',r}p(s',r|s,\pi(s))[r+\gamma v_k(s')] \\
\end{align}
$$
在保证$v_\pi$存在的情况下，$v_{k}$会收敛刀$v_{\pi}$
### 策略改进
通过策略评估得到给定策略的价值函数之后。一种改进方法是，在状态$s$选择动作$a$后，继续遵循现有的策略$\pi$。
$$
\begin{align}
q_{\pi}(s,a)=&E[R_{t+1}+\gamma v_{\pi}(S_{t+1})|S_t=s,A_t=a]\\
=&\sum_{s',r}p(s',r|s,a)[r+\gamma v_{\pi}(s')]
\end{align}
$$
- 定理: $\pi,\pi '$式任意两个给定的策略，则$\pi '$优于$\pi$,若对任意$s \in S$，
$$
q_{\pi}(s,\pi '(s)) \geq v_{\pi}(s)
$$
- 一种策略改进方法：当前步使用如下策略
$$
\begin{align}
\pi '(s)=&{argmax}_{a}q_{\pi}(s,a)
\end{align}
$$
### 伪代码
```pseudo
\begin{algorithm}
\caption{PolicyIteration}
\begin{algorithmic}
  \PROCEDURE{Quicksort}{$A, p, r$}
	\IF{$p < r$}
	  \STATE $q = $ \CALL{Partition}{$A, p, r$}
	  \STATE \CALL{Quicksort}{$A, p, q - 1$}
	  \STATE \CALL{Quicksort}{$A, q + 1, r$}
	\ENDIF
  \ENDPROCEDURE
  \PROCEDURE{Partition}{$A, p, r$}
	\STATE $x = A[r]$
	\STATE $i = p - 1$
	\FOR{$j = p$ \TO $r - 1$}
	  \IF{$A[j] < x$}
		\STATE $i = i + 1$
		\STATE exchange
		$A[i]$ with $A[j]$
	  \ENDIF
	\STATE exchange $A[i]$ with $A[r]$
	\ENDFOR
  \ENDPROCEDURE
  \end{algorithmic}
\end{algorithm}
```
## 值迭代
算法优化
- 策略评估需要迭代那么多次吗？
Any policy evaluation algorithm
Any policy improvement algorithm
#### Bellman Optimality Equation
# 蒙特卡洛