import numpy as np
import matplotlib.pyplot as plt

# 定义几种常见激活函数
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def swish(x):
    return x * sigmoid(x)

# 生成输入数据：模拟神经网络某一层的线性变换输出
x = np.linspace(-5, 5, 1000)

# 应用激活函数
activations = {
    "ReLU": relu(x),
    "Sigmoid": sigmoid(x),
    "Tanh": tanh(x),
    "Leaky ReLU": leaky_relu(x),
    "Swish": swish(x)
}

# 画出每个激活函数的输出值域分布（histogram）
for name, act in activations.items():
    plt.figure()
    plt.hist(act, bins=50, edgecolor='black')
    plt.title(f"{name} Activation Output Distribution")
    plt.xlabel("Activation Output")
    plt.ylabel("Frequency")
    plt.grid(True)

plt.show()
