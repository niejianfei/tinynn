import numpy as np
from initializer import zeros_init
from initializer import normal_init


# model进行forward和backward，其实底层都是网络层在进行实际运算，因此网络层需要有提供forward和backward接口进行对应的运算。
# 同时还应该将该层的参数和梯度记录下来
class Layer(object):
    def __init__(self, name):
        self.name = name
        # params初始化获取，grads反向传播获取
        self.params, self.grads = None, None

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


"""
全连接层：
forward
k -> k+1层
activation(X(k) * W(K+1) + b(k+1)) = X(k+1)
batch_size * num_inputs => num_inputs * num_outputs + 1 * num_outputs => activation => batch_size * num_outputs

backward
k+1 -> k层:
对W导数：X(k).T * grad(X(k+1)), (num_inputs * batch_size) * (batch_size * num_outputs)
对b导数: grad(X(k+1)), 1 * num_outputs
"""


class Dense(Layer):
    def __init__(self, num_in, num_out,
                 w_init=normal_init,
                 b_init=zeros_init):
        super().__init__("Linear")

        self.params = {
            "w": w_init((num_in, num_out)),
            "b": b_init((1, num_out)).reshape(-1)
        }

        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad):
        self.grads["w"] = self.inputs.T @ grad
        self.grads["b"] = np.sum(grad, axis=0)
        return grad @ self.params["w"].T


# 激活函数可以看做是一种网络层，同样需要实现forward和backward方法。
class Activation(Layer):
    """Base activation layer"""

    def __init__(self, name):
        # super()函数是用来调用父类的方法的,它可以让子类调用父类的方法,从而实现代码的复用
        # 在调用super()函数时，若传递self和name作为参数。由于super()函数中的第一个参数是类，而不是实例，因此会引发Unexpected argument错误。
        super().__init__(name)
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

    def backward(self, grad):
        return self.derivative_func(self.inputs) * grad

    def func(self, x):
        raise NotImplementedError

    def derivative_func(self, x):
        raise NotImplementedError


class ReLU(Activation):
    """ReLU activation function"""

    def __init__(self):
        super().__init__("ReLU")

    def func(self, x):
        return np.maximum(x, 0.0)

    def derivative_func(self, x):
        # true:1, false:0
        return x > 0


if __name__ == '__main__':
    dense = Dense(12, 36)
    print(dense.params, dense.grads)

    x = np.random.normal(0, 1, (3, 3))
    relu = ReLU().func(x)
    print(relu)
