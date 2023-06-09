定性变量与定量变量的归一化  

| ID  | outlook | temperature |
| --- | ------- | ---------- |
| 1   | Sunny     | 80         |
| 2   | rainy    | 64         |
| 3   | overcast  | 70         |    

x1,x2的距离，不同距离下变量对应归一化尺度  

| outlook                       | temperature |
| ----------------------------- | ----------- |
| \|1-0\|+\|0-1\|+\|0-0\|=2 |  归一化到$[-1,1]$  |
| $\sqrt{(1-0)^2+(0-1)^2+(0-0)^2}=\sqrt{2}$ |  归一化到$[-\sqrt{2},\sqrt{2}]$  |

## 数据清洗
### 缺失值插补 
回归与随机回归插补的问题：
1) 不能使用含缺失值的变量作为预测变量
2) 插补的值可能越出已观测值范围（回归中以与预测值最接近的观测值插补，随机回归中截断）
#### 回归插补
以需要插补的变为目标，其他相关变量为预测变量建立模型，以预测值为插补值。  
注意事项：
1) 需要对每一变量独立进行插补（不能使用有缺失的样本观测信息）
#### 随机回归插补
#### 缺失森林

1) 冷卡插补
2) 非条件中心插补
3) 条件分层后中心插补
4) K近邻插补
5) 回归插补
6) 随机回归插补
7) 缺失森林插补
8) MICE
9) 预测均值匹配
## 数据规约
### 特征选择
#### 过滤法
#### 封装法
##### 递归特征删除
加入或删除的特征不能更改（不参与后续的步骤）
1) 向前，每次加入1个（基于交叉验证）
2) 向后，每次剔除1个
3) 增l删r，  
特点：
1) 不能修改前面已加入或删除特征的操作
2) 操作是强制的不能根据效果更改（即使加入特征模型效果变差也加入变量）
3) 未能穷举所有特征组合，无法保证局部最优
### 特征提取
#### 投影寻踪
基本思想：通过极大化（极小化）选定的投影指标，寻找能够反映原始数据结构或特征的投影方向。
##### 投影指标
1) 针对聚类：Friedman指标和最小化峰度指标
2) Friedman-Tukey指标和最大化峰度指标较好，Friedman也可以 
###### Friedman-Tukey
窗口$R$表示，用距离小于$R$的邻居计算密度。
###### Friedman
测量偏离正态分布的程度(因为正态不好，单峰不好分类等)，最大化Friedman投影指标（越偏离正态）

$$
Y=2g(Z)-1
$$

$g()$为标准正态分布函数。 

$$
I_F(\alpha)=\int_{-1}^1(f(y)-0.5)^2dy
$$

$f(y)$是Y的密度函数，因为积分计算困难可以做正交多项式展开。
