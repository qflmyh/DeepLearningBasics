import torch
import torch.nn as nn
from torchsummary import  summary #计算模型参数，查看模型结构

"""
深度学习4个步骤
1.准备数据
2.搭建神经网络
3.模型训练
4。模型测试

搭建神经网络流程：
1.定义一个类，继承nn.Module
2.在__init__()方法中搭建神经网络
3.在forward()方法中完成前向传播
"""

# 搭建神经网络
class ModeDemo(nn.Module):
    def __init__(self):
        # 初始化父类成员
        super().__init__()

        # 搭建神经网络：隐藏层 + 输出层
        # 定义全连接层（线性层）：nn.Linear(输入特征数, 输出特征数)
        # 隐层层1：输入特征数3，输出特征数3
        self.linear1 = nn.Linear(3, 3)
        # 隐层层2：输入特征数3，输出特征数2
        self.linear2 = nn.Linear(3, 2)
        # 输出层：输入特征数2，输出特征数2
        self.output = nn.Linear(2, 2)

        # 对隐藏层进行参数初始化
        # xavier_uniform_：针对 Sigmoid / Tanh等 “对称激活函数”，保证前向 / 反向传播的梯度方差一致；
        # kaiming_normal_（He初始化）：针对ReLU等 “非对称激活函数”，解决梯度消失问题；
        # zeros_：偏置初始化为0是常见做法（也可初始化为小常数）。
        # 隐藏层1
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        # 隐藏层2
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)


    def forward(self, x):
        # 第1层:隐藏层计算：加权求和 + 激活函数(Sigmoid)
        x = torch.sigmoid(self.linear1(x))
        # 第2层:隐藏层计算：加权求和 + 激活函数(Relu)
        x = torch.relu(self.linear2(x))
        # 第3层:输出层计算：加权求和 + 激活函数(Softmax)
        x = torch.softmax(self.output(x), dim=-1)  #dim=-1表示按行计算

        return x

# 模型训练
def train():
    # 创建模型对象
    model = ModeDemo()

    # 创建数据集样本，随机生成
    data = torch.randn(size=(5,3))
    print(f'data: {data}')
    print(f'data.shape: {data.shape}')
    print(f'data.requires_grad: {data.requires_grad}')          #False

    # 调用神经网络模型 --> 模型训练
    output = model(data)        #地层自动调用forward()方法，进行前向传播
    print(f'output: {output}')
    print(f'output.shape: {output.shape}')                      #5行2列
    print(f'output.requires_grad: {output.requires_grad}')      #True

    # 计算和查看模型参数
    print("====================计算模型参数=======================")
    # 参1：神经网络模型对象 参2：输入数据的维度
    summary(model, input_size=data.shape)

    print("====================查看模型参数========================")
    for name, param in model.named_parameters():
        print(f'name: {name}, param: {param}')

if __name__ == '__main__':
    train()

