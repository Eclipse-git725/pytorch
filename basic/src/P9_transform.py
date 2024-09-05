from torch.utils.tensorboard import SummaryWriter 
from torchvision import transforms
from PIL import Image


img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")


# 如何使用transforms
# tensor_img = transforms.ToTensor()(img)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)

writer.add_image("Tensor_img", tensor_img)

writer.close()