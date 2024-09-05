import torch
import torchvision
from torch import nn

# 太大了，147G
# dataset = torchvision.datasets.ImageNet("./dataset", split="train", transform=torchvision.transforms.ToTensor(), download=True)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

vgg16_false = torchvision.models.vgg16(pretrained=False)
# vgg16_true = torchvision.models.vgg16(pretrained=True)
# 调试，断点打到这里
print("ok")
print(vgg16_false)

# 迁移学习，根据现有网络改变它的结构
# 修改vgg16，in_features=1000，out_features=10
# vgg16_false.add_module("add_linear", nn.Linear(1000, 10))
# vgg16_false.classifier.add_module("add_linear", nn.Linear(1000, 10))
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)