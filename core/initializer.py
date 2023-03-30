import numpy as np


def normal_init(shape):
    """normal init"""
    return np.random.normal(loc=0, scale=1, size=shape)


def zeros_init(shape):
    """zero init"""
    return np.zeros(shape)


if __name__ == '__main__':
    shape1 = (12, 784)
    init = zeros_init
    print(init)
    print(type(init))
    print(init(shape1))
