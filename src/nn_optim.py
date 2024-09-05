from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # sequential的使用
        self.model1 = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
        
    def forward(self, x):
        x = self.model1(x)
        return x
    
dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

# writer = SummaryWriter("logs")
model = Model()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(3):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        output = model(imgs)
        result_loss = loss(output, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss += result_loss
    print(f"epoch: {epoch}, loss: {running_loss}")
