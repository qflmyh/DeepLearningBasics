"""
    CNN卷积神经网络————图像分类案例
    深度学习项目步骤
        1.准备数据集：
            用计算机视觉模块torchvision自带的CIFAR10数据集，包含6W张（32，32，3）的图片，5W张训练集，1W张测试集，10个分类，每个分类6K张图片
        2.搭建卷积神经网络
        3.模型训练
        4.模型测试

        卷积层：
            提取图像的局部特征 -> 特征图（Feature Map）,计算方式: N = (W - F + 2*P) / S   +  1
            每个卷积核都是一个神经元
        池化层：
            降维，有最大池化和平均池化
            池化只在HW上做调整，通道上不改变。

        案例的优化思路:
            1. 增加卷积核的输出通道数(大白话: 卷积核的数量)
            2. 增加全连接层的参数量.
            3. 调整学习率
            4. 调整优化方法(optimizer...)
            5. 调整激活函数...
            6. ...
"""

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

#每批次样本数
BATCH_SIZE = 8

# 1.准备数据集
def creat_dataset():
    # 获取训练集
    # 参1：数据集路径 参2：是否是训练集 参3：数据预处理 -> 张量数据 参4：是否联网下载
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
    # 获取测试集
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
    # 返回数据集
    return train_dataset, test_dataset

# 2.搭建卷积神经网络
class CNNImageModel(nn.Module):
    # 初始化父类成员，搭建神经网络
    def __init__(self):
        # 初始化父类成员
        super().__init__()
        # 搭建神经网络
        # 第1个卷积层，输入3通道，输出6通道，卷积核大小3*3，步长1，填充0
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        # 第1个池化层，窗口大小2*2，步长2，填充0
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 第2个卷积层，输入6通道，输出16通道，卷积核大小3*3，步长1，填充0
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)
        # 第2个池化层，窗口大小2*2，步长2，填充0
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 全连接层
        # 第1个隐藏层，输入576，输出120
        self.linear1 = nn.Linear(in_features=16*6*6, out_features=120)
        # 第2个隐藏层，输入120，输出84
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        # 第3个隐藏层 -> 输出层，输入84，输出10
        self.output = nn.Linear(in_features=84, out_features=10)

    # 定义正向传播
    def forward(self, x):
        # 第1层：卷积层（加权求和）+ 激励层（激活函数）+ 池化层（降维）
        x = self.pool1(torch.relu(self.conv1(x)))
        # 第2层：卷积层（加权求和）+ 激励层（激活函数）+ 池化层（降维）
        x = self.pool2(torch.relu(self.conv2(x)))

        # 全连接层只能处理二维数据，所以要拉平数据（8，16，6，6）-> (8,576)
        # 参1：样本数（行数） 参2：列数（特证数） -1表示自动计算
        x = x.reshape(x.size(0), -1)   #8行576列
        # print(f'x.shape: {x.shape}')

        # 第3层：全连接层（加权求和）+ 激励层（激活函数）
        x = torch.relu(self.linear1(x))
        # 第4层：全连接层（加权求和）+ 激励层（激活函数）
        x = torch.relu(self.linear2(x))
        # 第5层：全连接层（加权求和） -> 输出层
        # 后续用多分类交叉熵损失函数CrossEntropyLoss = softmax()激活函数 + 损失计算
        return self.output(x)

# 3.模型训练
def train(train_dataset):
    # 创建数据加载器
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 创建模型对象
    model = CNNImageModel()
    # 创建损失函数对象
    criterion = nn.CrossEntropyLoss()   #多分类交叉熵损失函数CrossEntropyLoss = softmax()激活函数 + 损失计算
    # 创建优化器对象
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 循环遍历epoch，开始每轮的训练
    # 定义训练的总轮数
    epochs = 10
    # 遍历，完成每轮批次的训练
    for epochs_idx in range(epochs):
        # 定义变量，记录：总损失，总样本数据量，预测正确样本个数，训练开始时间
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
        # 遍历数据加载器，获取每批次的数据
        for x, y in dataloader:
            # 切换训练模式
            model.train()
            # 模型预测
            y_pred = model(x)
            # 计算损失
            loss = criterion(y_pred, y)
            # 梯度清零 + 反向传播 + 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计预测正确的样本个数
            # print(y_pred)        预测概率
            # argmax()返回最大值对应的索引，即该图片的预测分类
            # print(torch.argmax(y_pred, dim=-1))  #-1表示行
            # print(y)              真实值
            # print(torch.argmax(y_pred, dim=-1) == y) 是否预测正确
            # print((torch.argmax(y, dim=-1) == y).sum())  预测正确的样本个数
            total_correct += (torch.argmax(y_pred, 1) == y).sum()
            # 统计当前批次的总损失
            total_loss += loss.item()*len(y)    # [第1批总损失 +  第2批总损失 +  第3批总损失 +  ...]
            # 统计当前批次的总样本数
            total_samples += len(y)

        # 打印该轮的训练信息
        print(f'epoch:{epochs_idx + 1},loss:{total_loss/total_samples:.5f},acc:{total_correct / total_samples:.2f}, time:{time.time()-start:.2f}s')

    # 保存模型
    torch.save(model.state_dict(), './model/CNNImage_model.pth')

# 4.模型测试
def evaluate(test_dataset):
    # 创建测试集数据加载器对象
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 创建模型对象
    model = CNNImageModel()
    # 加载模型参数
    model.load_state_dict(torch.load('./model/CNNImage_model.pth'))
    # 定义变量统计预测正确的样本个数，总样本个数
    total_correct, total_samples = 0, 0
    # 遍历数据加载器，获取每批次的数据
    for x, y in dataloader:
        # 切换模型模式
        model.eval()
        # 模型预测
        y_pred = model(x)
        # 训练时用了CrossEntropyLoss,所以搭建神经网络时没有加softmax激活函数，这里用argmax函数模拟
        # argmax()返回最大值对应的索引，即该图片的预测分类
        # 统计预测正确的样本数
        total_correct += (torch.argmax(y_pred, dim=-1) == y).sum()  #-1表示行
        # 统计总样本个数
        total_samples += len(y)

    # 打印正确率（预测结果）
    print(f'Accuracy: {total_correct / total_samples:.2f}')

#5.测试
if __name__ == "__main__":
    # 1.准备数据集
    train_dataset, test_dataset = creat_dataset()
    # print(f'train_dataset: {train_dataset.data.shape}') #(50000,32,32,3)
    # print(f'test_dataset: {test_dataset.data.shape}')   #(10000,32,32,3)
    # # 数据集类别: {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    # print(f'数据集类别: {train_dataset.class_to_idx}')

    # # 图像展示
    # plt.figure(figsize=(2, 2))
    # plt.imshow(train_dataset.data[1111])      # 索引为1111的图像
    # plt.title(train_dataset.targets[1111])
    # plt.show()

    # 2.搭建卷积神经网络
    # model = CNNImageModel()
    # # 查看模型参数 参1：模型 参2：输入维度（CHW，通道，高，宽），参3：批次大小
    # summary(model, input_size=(3, 32, 32), batch_size=BATCH_SIZE)

    # 3.模型训练
    # train(train_dataset)
    # 4.模型测试
    evaluate(test_dataset)
