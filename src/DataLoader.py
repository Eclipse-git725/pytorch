from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision

# 准备的测试数据集
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
# drop_last=True: 如果最后一个batch的数据量小于batch_size，就把这个batch丢掉
# shuffle=True: 每个epoch都打乱数据
test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter("logs")

# 测试数据中的第一张图片和target
img, target = test_set[0]
print(img.shape)
print(target)

for epoch in range(2):                                                                                                            
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        step += 1

writer.close()