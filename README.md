
<!-- README.md is generated from README.Rmd. Please edit that file -->


# 果蝇优化算法
基本思想
果蝇优化算法是受到果蝇觅食行为启发而发展的。在这种算法中，果蝇通过嗅觉去寻找食物，而食物的位置相当于问题的最优解。

主要组成部分
信息素（Smell）：用于表示目标函数值或适应度值。
位置（Position）：代表搜索空间中的一个点，对应一个可能的解。
步骤
初始化果蝇群体的位置。
在每一次迭代中：
评估每个果蝇的信息素浓度（即目标函数的值）。
找出信息素浓度最小（或最大）的果蝇，即找到当前最优的解。
根据当前最优解更新其他果蝇的位置。

## Usage

``` python
def fruitfly(objective_func, n, max_gen):
    # n: 种群规模
    # max_gen: 最大迭代次数
    Smell = np.zeros(n)
    x = np.random.rand(n)
    best = np.inf
    best_x = None

    for gen in range(max_gen):
        for i in range(n):
            Smell[i] = objective_func(x[i])

        min_smell = np.min(Smell)
        if min_smell < best:
            best = min_smell
            best_x = x[np.argmin(Smell)]

        x = best_x + np.random.randn(n) * (max_gen - gen) / max_gen

    return best_x, best
def objective_func(x):
    return x ** 2
best_x, best_value = fruitfly(objective_func, 20, 100)
```
在给出的代码中，`objective_func` 是目标函数（也叫做适应度函数或成本函数），它用于评估算法寻找的解的质量。该函数接受一个解（在这个例子中是一个数值 `x`）作为输入，然后返回一个标量值，表示该解的适应度或成本。

具体到这个例子：

```python
def objective_func(x):
    return x ** 2
```

`objective_func` 是一个简单的平方函数，用于评估解 `x` 的质量。算法的目标是找到使这个函数值最小化的 `x` 值。在这个特定例子中，最优解显然是 `x = 0`，此时 `objective_func(x)` 的值也是最小的，即 0。

在果蝇优化算法中，每一只“果蝇”都有一个位置 `x[i]`，这个位置代表了一个潜在的解。通过应用 `objective_func` 函数到这个位置上，我们可以得到一个“气味”或“信息素浓度”（在代码中存储在数组 `Smell` 中），这反映了该位置（解）的质量。算法的目标是通过不断迭代更新果蝇的位置，最终找到一个最优（或接近最优）的解。

简单来说，`objective_func` 就是用来衡量解好坏的标准。在不同的优化问题中，这个目标函数可以是非常复杂的，可能包括多个变量和约束条件。

# 粒子群算法
粒子群优化算法（PSO）
基本思想
粒子群优化算法是模仿鸟群或鱼群等动物群体的集体行为而发展起来的。算法中每一个解（一组参数）被视为“空间”中的一个“粒子”。这些粒子在优化的过程中，根据自己的经验和群体的经验来调整自己的位置。

主要组成部分
粒子（Particle）：代表在搜索空间中的一个可能解。
速度（Velocity）：控制粒子移动到新位置的速率和方向。
位置（Position）：粒子在搜索空间中的具体位置，代表一个可能解。
步骤
初始化粒子群的位置和速度。
在每一次迭代中：
评估每个粒子的适应度（即目标函数的值）。
更新每个粒子的个体最优解和全局最优解。
根据个体最优解和全局最优解更新粒子的速度和位置。


## Usage

``` python
from pyswarm import pso

def my_function(x):
    x1, x2 = x
    return x1**2 + x2**2

lb = [-10, -10]  # 下界
ub = [10, 10]  # 上界

xopt, fopt = pso(my_function, lb, ub)
```
