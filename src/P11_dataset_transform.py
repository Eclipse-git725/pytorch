from torch.utils.tensorboard import SummaryWriter
import torchvision

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
writer = SummaryWriter("logs")

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)

# print(train_set[0])
# print(train_set.classes)
# img, target = train_set[0]
# print(img)
# print(target)
# img.show()

print(test_set[1])
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()