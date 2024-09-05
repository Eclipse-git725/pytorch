import torch
import torchvision
from torch import nn
from model_save import *

# 读取方式1
model = torch.load('vgg16_method1.pth')
# print(model)

# 读取方式2
model = torch.load('vgg16_method2.pth')
# print(model)
# 只保存参数，如何恢复网络模型，load_state_dict参数里写字典
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load('vgg16_method2.pth'))
# print(vgg16)

# 陷阱
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv2d1 = nn.Conv2d(3, 64, 3)

#     def forward(self, x):
#         return self.conv2d1(x)
# 只写load会报错，需要把模型定义写上，或者用import引入
model = torch.load('model_method1.pth')
print(model)