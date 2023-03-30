from layer import Dense
from layer import ReLU

"""
net 类负责管理 tensor 在 layers 之间的前向和反向传播。forward方法，按顺序遍历所有层，每层计算的输出作为下一层的输入；backward 则逆序遍历所有层，将每层的梯度作为下一层的输入。
这里还将每个网络层参数的梯度保存下来返回，后面参数更新需要用到。另外 net 类还实现了获取参数、设置参数、获取梯度的接口，也是后面参数更新时需要用到
"""


class Net(object):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        all_grads = []
        for layer in reversed(self.layers):
            # 更新梯度
            grad = layer.backward(grad)
            all_grads.append(layer.grads)
        return all_grads[::-1]  # 列表逆序

    def get_params_and_grads(self):
        for layer in self.layers:
            yield layer.params, layer.grads

    def get_parameters(self):
        return [layer.params for layer in self.layers]

    def set_parameters(self, params):
        for i, layer in enumerate(self.layers):
            for key in layer.params.keys():
                layer.params[key] = params[i][key]


if __name__ == '__main__':
    net_list = [Dense(784, 256), ReLU(), Dense(256, 10), ReLU()]
    net = Net(net_list)
    print(net.get_parameters()[1])

    print(type(net.get_params_and_grads()))
    print(next(net.get_params_and_grads()))

    for i, j in enumerate(net.get_params_and_grads()):
        print(f"第{i+1}层：", j)
