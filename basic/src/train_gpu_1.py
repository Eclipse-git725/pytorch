from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader
from torch import nn
import torch
import time

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

print("训练数据集的长度为：{}".format(len(train_set)))
print("测试数据集的长度为：{}".format(len(test_set)))

# 用DataLoader加载数据集
train_dataloader = DataLoader(train_set, batch_size=64, drop_last=True)
test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)

# 搭建神经网络
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)
    
model = Model()
if torch.cuda.is_available():
    model.cuda()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn.cuda()

# 创建优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 记录训练轮数
epoch = 5

# 添加tensorboard
writer = SummaryWriter("logs")


for i in range(epoch):
    print("-----------第{}轮训练开始---------".format(i+1))

    # 训练步骤开始
    # 对特殊的层有影响，比如dropout和batchnorm
    model.trian()
    start_time = time.time()
    for data in train_dataloader:
        imgs, target = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            target = target.cuda()
        output = model(imgs)
        loss = loss_fn(output, target)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if(total_train_step % 100 == 0):
            print("训练次数：{}，loss：{}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    end_time = time.time()
    print("第{}轮训练结束，耗时：{}".format(i+1, end_time-start_time))
    # 测试步骤开始
    # 对特殊的层有影响，比如dropout和batchnorm
    model.eval()
    total_test_loss = 0
    # 分类中的重要参数，正确率
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, target = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                target = target.cuda()
            output = model(imgs)
            loss = loss_fn(output, target)
            total_test_loss += loss
            # 计算正确率
            accuracy = (output.argmax(1) == target).sum()
            total_accuracy += accuracy

    print("整体测试集上的loss为：{}".format(total_test_loss))
    print("整体测试集上的正确率为：{}".format(total_accuracy/len(test_set)))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/len(test_set), total_test_step)
    total_test_step += 1

    # 保存每一轮训练的模型
    # torch.save(model, "model/model_{}.pth".format(i))
    # print("模型已保存")

writer.close()