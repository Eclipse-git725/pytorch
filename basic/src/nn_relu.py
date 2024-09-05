from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.nn import ReLU
from torch.nn import Sigmoid



input = torch.tensor([[1, -1.5],
                      [-1,3]])
input = torch.reshape(input, (1, 2, 2))

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
writer = SummaryWriter("logs")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        
    def forward(self, x):
        # output = self.relu(x)
        output = self.sigmoid(x)
        return output

model = Model()
# output = model(input)
# # tensor([[[1., 0.],[0., 3.]]])
# print(output)

step = 0
for data in dataloader:
    imgs, targets = data
    output = model(imgs)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step += 1

writer.close()