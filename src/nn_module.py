from torch import nn
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        output = input + 1
        return output
    
model = Model()
input = torch.tensor(1)
output = model(input)
print(output) 