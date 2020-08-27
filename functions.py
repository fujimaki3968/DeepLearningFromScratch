import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# p69
def softmax(x):
    C = np.max(x)
    # 誤差を減らすぞい
    exp_x = np.exp(x - C)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

