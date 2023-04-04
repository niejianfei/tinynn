import numpy as np


# losses 组件需要做两件事情，给定了预测值和真实值，需要计算损失值和关于预测值的梯度。我们分别实现为 loss 和 grad 两个方法

class BaseLoss(object):

    def loss(self, predicted, actual):
        raise NotImplementedError

    def grad(self, predicted, actual):
        raise NotImplementedError
