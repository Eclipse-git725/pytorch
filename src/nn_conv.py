import torch
import torch.nn.functional as F

# 5x5的输入
input = torch.tensor([[1, 2, 0, 3, 1], 
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

# 3x3的卷积核
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

print(input.shape)
print(kernel.shape)

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape)
print(kernel.shape)

# stride的作用是卷积核每次移动的步长
output = F.conv2d(input, kernel, stride=1)
print(output)

output2 = F.conv2d(input, kernel, stride=2)
print(output2)

# padding的作用是在输入的边缘补0
output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)