import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset             #数据集对象， 数据 ->Tensor->数据集->数据加载器
from torch.utils.data import DataLoader                #数据加载器
import torch.nn as nn                                  #neural network神经网络
import torch.optim as optim                            #优化器
from sklearn.model_selection import train_test_split   #训练集与测试集的划分
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time                                            #时间模块
from torchsummary import summary                       #模型结构可视化

"""
    ANN人工神经网络 ————手机价格分类案例
    基于手机的20项特征 -> 预测手机的价格区间

    ANN实现步骤：
        1.构建数据集
        2.搭建神经网络
        3.模型训练
        4.模型测试
    
    优化思路：
        1.优化方法SGD -> Adam
        2.学习率从0.01降为0.001
        3.对数据进行标准化
        4.增加网络深度，调整神经元深度
        5.增加训练轮数
"""

# 1.构建数据集
def create_dataset():
    # 读取数据文件
    data = pd.read_csv('./data/手机价格预测.csv')
    # print(f'data: {data.head()}')
    # print(f'data: {data.shape}')      (2000,21)

    # 获取x特征列，y标签列
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    # print(f'x: {x.head(), {x.shape}}')  (2000, 20)
    # print(f'y: {y.head()}, {y.shape}')    (2000, )

    # 特征列转成浮点型
    x = x.astype(np.float32)
    # print(f'x: {x.head(), {x.shape}}')  (2000, 20)

    # 划分训练集和测试集
    # 参1：特征，参2：标签，参3：测试集所占比例，参4：随机种子，参5：样本的分布（参考y的类别抽取数据）
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=88, stratify=y)
    # 数据标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 数据集封装为张量数据集     数据 -> 张量Tensor -> 数据集TensorDataSet -> 数据加载器DataLoader
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test.values))
    # print(f'train_dataset: {train_dataset}, test_dataset: {test_dataset}')

    # 返回结果                              20(充当输入特征数)  4（充当输出标签数）
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))

# 2.搭建神经网络
class ANNPhonePriceModel(nn.Module):
    # 初始化父类成员及搭建神经网络
    def __init__(self, input_dim, output_dim):  #输入20 输出4
        # 初始化父类成员
        super().__init__()
        # 搭建神经网络
        # 隐藏层1
        self.linear1 = nn.Linear(input_dim, 128)
        # 隐藏层2
        self.linear2 = nn.Linear(128, 256)
        # 输出层
        self.output = nn.Linear(256, output_dim)

    # 定义正向传播方法forward()
    def forward(self, x):
        # 隐藏层1：加权求和 + 激活函数（relu）
        x = torch.relu(self.linear1(x))
        # 隐藏层2：加权求和 + 激活函数（relu）
        x = torch.relu(self.linear2(x))
        # 隐藏层3：加权求和 + 激活函数（softmax） -> 这里只需要加权求和
        # x = self.softmax(self.output(x), dim=1)为正常写法，但不需要，后续用多分类交叉熵损失函数 CrossEntropyLoss()替代
        # CrossEntropyLoss() = softmax() + 损失计算
        x = self.output(x)
        # 返回处理结果
        return x

# 3.模型训练
def train(train_dataset,input_dim, output_dim):
    # 创建数据集加载器
    # 数据集训练（1600条）    参2：每批次的数据集条数，参3：是否打乱数据（训练集打乱，测试集不打乱）
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # 创建神经网络模型
    model = ANNPhonePriceModel(input_dim, output_dim)
    # 定义损失函数，因为是多分类，用多分类交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 创建优化器对象
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # 模型训练
    # 定义变量记录训练总轮数
    epochs = 50
    # 开始每轮训练
    for epoch in range(epochs):
        # 记录每次训练损失值和训练批次数
        total_loss, batch_num = 0, 0
        # 记录训练开始时间
        start_time = time.time()
        # 开始本轮各个批次训练
        for x, y in train_loader:
            # 切换模型（状态）
            model.train()  #训练模式  model.eval() #测试模式
            # 模型预测
            y_pred = model(x)
            # 计算损失
            loss = criterion(y_pred, y)
            # 梯度清零，反向传播，优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 累加损失值
            # 把本轮的每批次（16条）平均损失累计起来
            total_loss += loss.item()
            batch_num += 1
        # 本轮训练结束，打印训练信息
        print(f'epoch: {epoch + 1}, loss: {total_loss / batch_num:.4f}, time: {time.time() - start_time:.2f}s')
    # 多轮训练结束，保存模型参数
    # 参1：模型对象的参数（权重矩阵，偏置矩阵） 参2：模型保存文件名
    # print(f'\n\n模型的参数信息: {model.state_dict()}\n\n')
    torch.save(model.state_dict(), './model/ANNPhonePriceModel.pth')    #后缀名pth,pkl,pickle均可

# 4.模型测试
def evaluate(test_dataset,input_dim, output_dim):
    # 创建神经网络分类对象
    model = ANNPhonePriceModel(input_dim, output_dim)
    # 加载模型参数
    model.load_state_dict(torch.load('./model/ANNPhonePriceModel.pth'))
    # 创建测试集的数据加载器对象
    # 数据集训练（400条）    参2：每批次的数据集条数，参3：是否打乱数据（训练集打乱，测试集不打乱）
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    # 定义变量记录预测正确的样本个数
    correct = 0
    # 从数据加载器中获取每批的数据
    for x, y in test_loader:
        # 切换模型状态 -> 测试模式
        model.eval()
        # 模型预测
        y_pred = model(x)
        # 根据加权求和，得到类别，用argmax()获取最大值对应的下标就是类别
        y_pred = torch.argmax(y_pred, dim=1) #dim=1表示按行处理
        # print(f'y_pred: {y_pred}')
        # print(f'y:{y}')
        # 统计预测正确的样本个数
        # print(y_pred == y)
        # print((y_pred == y).sum())
        correct += y_pred.eq(y).sum().item()

    # 打印准确率
    print(f'准确率(accuracy={correct / len(test_dataset):.4f})')

# 5.测试
if __name__ == '__main__':
    # 1.构建数据集
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    # print(f'train_dataset: {train_dataset}')
    # print(f'test_dataset: {test_dataset}')
    # print(f'input_dim: {input_dim}')    #20
    # print(f'output_dim: {output_dim}')  #4

    # 2.搭建神经网络
    # model = ANNPhonePriceModel(input_dim, output_dim)
    # 计算模型参数
    # 参1：模型对象 参2：输入数据的形状（批次大小， 输入特征数），每批16条，每条20列特征
    # summary(model, input_size=(16, input_dim))

    # 3.模型训练
    train(train_dataset, input_dim, output_dim)
    # 4.模型测试
    evaluate(test_dataset, input_dim, output_dim)