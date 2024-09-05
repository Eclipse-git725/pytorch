from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader

# 5x5的输入
input = torch.tensor([[1, 2, 0, 3, 1], 
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])
input = input.reshape(1, 5, 5)

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataload = DataLoader(dataset, batch_size=64, shuffle=True)
# writer = SummaryWriter("logs")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=True)
        
    def forward(self, x):
        output = self.maxpool(x)
        return output
    
model = Model()
output = model(input)
print(input.shape)
print(output)

step = 0
for data in dataload:
    imgs, targets = data
    # print(imgs.shape)
    # output = model(imgs)
    # writer.add_images("input", imgs, step)
    # writer.add_images("output", output, step)
    step += 1

# writer.close()