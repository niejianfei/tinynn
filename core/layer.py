import numpy as np

'''model进行forward和backward，其实底层都是网络层在进行实际运算，因此网络层需要有提供forward和backward接口进行对应的运算。同时还应该将该层的参数和梯度记录下来。'''


class Layer(object):
    def __int__(self, name):
        self.name = name
        self.params, self.grad = None, None

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, outputs):
        raise NotImplementedError


