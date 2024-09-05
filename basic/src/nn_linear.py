import torchvision
from torch.utils.data import DataLoader
from torch import nn
import torch

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(196608, 10)
        
    def forward(self, x):
        output = self.linear(x)
        return output
    
model = Model()
for data in dataloader:
    img, taget = data
    print(img.shape)
    # output = torch.reshape(img, (1, 1, 1, -1))
    output = torch.flatten(img)
    print(output.shape)
    output = model(output)
    print(output.shape)