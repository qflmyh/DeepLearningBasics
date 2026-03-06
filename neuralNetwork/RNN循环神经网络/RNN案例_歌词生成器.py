"""
    RNN循环神经网络————歌词生成器案例
        基于周杰伦歌词来训练模型，用给定的起始词，结合长度，生成AI歌词

    实现步骤；
        1.获取数据，进行分词，获取词表
        2.数据预处理，构建数据集
        3.搭建RNN神经网络
        4.模型训练
        5.模型预测
"""

import torch
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time

# 1.获取数据，进行分词，获取词表
def build_vocab():
    # 定义变量记录：去重后所有的词，每行文本分词结果
    unique_words, all_words = [], []
    # 遍历数据集，获取每行文本
    for line in open('./data/jaychou_lyrics.txt', 'r', encoding='utf-8'):
        # 获取每行歌词，进行分词
        words = jieba.cut(line)
        # 所有分词结果记录到all_words中
        all_words.append(words)        # [['想要', '有', '直升机', '\n'], [第2句歌词切词], ......]
        # 遍历分词结果，去重后添加到unique_words中
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    # 统计语料（去重后）中词的数量
    word_count = len(unique_words)      #5703个词
    # 构建词表，字典形式，key是词，value是词的索引
    # 例如:  {'想要': 0, '有': 1, '直升机': 2, '\n': 3, ...'冠军': 5701, '要大卖': 5702}
    word_to_index = { word : i for i, word in enumerate(unique_words) }
    # 歌词文本用词表索引表示
    corpus_idx = []
    # 遍历每一行的分词结果
    for words in all_words:
        # 定义变量记录词索引列表
        tmp = []
        # 获取每一行的词并获取相应的索引
        for word in words:
            tmp.append(word_to_index[word])
        # 在每行词之间添加空格隔开
        tmp.append(word_to_index[' '])
        # 获取文档中每个词的索引，添加到corpus_idx中
        corpus_idx.extend(tmp)
    # 返回结果：唯一词列表(5703个词), 词表 {'想要': 0, '有': 1, ... '要大卖': 5702},  (去重后)词的数量, 歌词文本用词表索引表示.
    return unique_words, word_to_index, word_count, corpus_idx

# 2.数据预处理，构建数据集
# 定义数据集类，继承torch.utils.data.Dataset
class LyricsDataset(torch.utils.data.Dataset):
    # 初始化词索引，词个数等……
    def __init__(self, corpus_idx, num_chars):
        # 文档数据中词的索引
        self.corpus_idx = corpus_idx
        # 每个句子中词的个数
        self.num_chars = num_chars
        # 文档数据中词的数量，不去重
        self.word_count = len(self.corpus_idx)
        # 句子数量
        self.number = self.word_count // self.num_chars

    # 当使用len(obj)时，自动调用此方法
    def __len__(self):
        # 返回句子数量
        return self.number

    # 当使用obj[index]时，自动调用此方法
    def __getitem__(self, idx):
        # idx：指的是词的索引，并将其修正索引值到文档的范围里面
        # 确定索引start在合法范围内，避免越界，start：当前样本的起始索引
        start = min(max(idx, 0), self.word_count - self.num_chars - 1)
        # 计算当前样本的结束索引
        end = start + self.num_chars
        # 输入值，从文档中取出start ~ end的索引的词 -> 作为x
        x = self.corpus_idx[start:end]
        # 输出值，网路预测结果
        y = self.corpus_idx[start + 1:end + 1]
        # 返回输入值和输出值 -> 张量形式
        return torch.tensor(x), torch.tensor(y)

# 3.搭建RNN神经网络
class RNNTextGenerator(nn.Module):
    # 初始化方法
    def __init__(self, unique_word_count):       # unique_word_count: 去重的词的数量(5703)
        # 初始化父类成员
        super().__init__()
        # 初始化词嵌入层：语料中词的数量， 词向量的维度
        self.ebd = nn.Embedding(unique_word_count, 128)
        # 循环网络层：词向量的维度，隐藏层的维度：256，网络层数1
        self.rnn = nn.RNN(128, 256, 1)
        # 输出层（全连接层）：特征向量维度（和隐藏层维度一致），词表中词的个数
        # 词表中每个词的概率 -> 选概率最大的那个词作为预测结果
        self.out = nn.Linear(256, unique_word_count)

    # 正向传播方法
    def forward(self, inputs, hidden):
        # 初始化词嵌入层处理
        # embd格式：（batch句子的数量，句子的长度，词向量维度）
        embd = self.ebd(inputs)

        # rnn处理
        # rnn格式：（句子的长度，batch句子的数量，隐藏层维度）
        output, hidden = self.rnn(embd.transpose(0,1), hidden)

        # 全连接，输入内容必须时二维数据，即词的数量 * 词的维度
        # 输入维度：（seq_len句子数量 * batch，词向量维度256）
        # 输出维度：（seq_len句子数量 * batch， 词表中词的个数）
        output = self.out(output.reshape(shape=(-1, output.shape[-1])))

        # 返回结果，预测结果，隐藏层
        return output, hidden

    # 隐藏层的初始化方法
    def init_hidden(self, bs):     #bs:batch_size
        # 隐藏层初始化：[网络层数，batch，隐藏层向量维度]
        return torch.zeros(1,bs,256)

# 4.模型训练
def train():
    # 构建词典
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    # 获取数据集
    lyrics = LyricsDataset(corpus_idx, 32)
    # 初始化神经网络模型
    model = RNNTextGenerator(unique_word_count)        #预测5703个词，每个词的概率

    # 创建数据加载器对象
    # 参1：数据集对象 参2：批次大小（每批5个句子，每个句子32个词） 参3：是否打乱数据
    lyrics_dataloader = DataLoader(lyrics, batch_size=5, shuffle=True)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 模型训练
    # 定义变量记录训练的轮数
    epochs = 10
    # 具体的每轮的训练动作
    for epoch in range(epochs):     # epoch: 0, 1, 2, 3...9, 分别表示: 第1轮, 第2轮, ... 第10轮.
        # 定义变量记录本轮开始训练时间，迭代（批次）次数，训练总损失
        start, iter_num, total_loss = time.time(), 0, 0.0

        # 具体的本轮各批次训练动作
        # 遍历数据集，后台会调用LyricsDataset #__getitem__()方法，获取每个样本的数据和标签
        for x, y in lyrics_dataloader:
            # 回去隐藏层初始值
            hidden = model.init_hidden(5)
            # 模型计算
            output, hidden = model(x, hidden)

            # 计算损失
            # y的形状：（batch批次数，seq_len句子长度, 词向量维度） -> 转成一维向量 -> 每个词的下标索引
            # output的形状：（seq_len，batch，词向量维度）
            y = torch.transpose(y, 0, 1).reshape(shape=(-1, ))
            loss = criterion(output, y)
            # 梯度清零 + 反向传播 + 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 累计损失和迭代次数
            total_loss += loss.item()
            iter_num += 1

        # 本轮训练结束，打印本轮训练信息
        print(f'epoch {epoch + 1}, time: {time.time() - start:.2f}s, loss {total_loss / iter_num:.4f}')

    # 模型训练结束，保存模型
    torch.save(model.state_dict(), './model/RNNTextGenerator.pth')

# 5.模型预测
def evaluate(start_word, sentence_length):
    # 构建词典
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    # 获取模型
    model = RNNTextGenerator(unique_word_count)
    # 加载模型参数
    model.load_state_dict(torch.load('./model/RNNTextGenerator.pth'))
    # 获取隐藏层初始值
    hidden = model.init_hidden(1)
    # 将输入的开始词转换成索引
    word_idx = word_to_index[start_word]
    # 定义列表，存放：产生的词的索引
    generated_sentence = [word_idx]     # 开始词的索引, 是列表的: 第1个值.
    # 遍历句子长度，获取到每一个词
    for i in range(sentence_length):
        # 模型预测
        output, hidden = model(torch.tensor([[word_idx]]), hidden)
        # 获取预测结果。 argmax()从所有结果（5703个词的概率）中，找最大值对应的索引
        word_idx = torch.argmax(output)
        # 把预测结果添加到列表中
        generated_sentence.append(word_idx)

    # 将索引转成词并打印
    for idx in generated_sentence:
        print(unique_words[idx], end='')

# 6.测试
if __name__ == '__main__':
    # 1.获取数据，进行分词，获取词表
    # unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    # print(f'词的数量word_count: {word_count}')
    # print(f'去重后的词·unique_words: {unique_words}')
    # print(f'每个词的索引word_to_index: {word_to_index}')
    # print(f'文档中每个词对应的索引corpus_idx: {corpus_idx}')

    # 2.数据预处理，构建数据集
    # dataset = LyricsDataset(corpus_idx, 5)
    # print(f'句子数量：{len(dataset)}')
    # # 查看输入值和目标值
    # x, y = dataset[1]
    # print(f'输入值x:{x}')
    # print(f'目标值y:{y}')

    # 3.搭建RNN神经网络
    # 创建模型对象
    # model = RNNTextGenerator(word_count)
    # # 查看参数
    # for name, parameter in model.named_parameters():
    #     print(f'参数名称name:{name}，参数维度parameter.shape:{parameter.shape}')

    # 4.模型训练
    # train()
    # 5.模型预测
    evaluate('星星', 10)