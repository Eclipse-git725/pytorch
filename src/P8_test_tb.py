from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")

# for i in range(100):
#     writer.add_scalar("y=2x", i * 3, i)

# writer.close()

img_path = "dataset/train/bees_image/16838648_415acd9e3f.jpg"
img = Image.open(img_path)
print(type(img))
numpy_img = np.array(img)
print(numpy_img.shape)
writer.add_image("test", numpy_img, 2, dataformats="HWC")

writer.close()