from PIL import Image
import torchvision
from torchvision import transforms
from torch import nn
import torch

img_path = "./imgs/dog.png"
img = Image.open(img_path)
print(img)

transform = torchvision.transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

img = transform(img)
print(img.shape)


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
    
model = torch.load("./model/model_4.pth", map_location=torch.device('cpu'))
print(model)
img = torch.reshape(img, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(img)
print(output)
print(output.argmax(1))