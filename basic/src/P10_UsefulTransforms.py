from torch.utils.tensorboard import SummaryWriter 
from PIL import Image
from torchvision import transforms

img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

print(tensor_img[0][0][0])
# Normalize使用
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norms =trans_norm(tensor_img)
print(img_norms[0][0][0])
writer.add_image("Normalize_img", img_norms)


# Resize使用
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> Resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> ToTensor -> img_resize tensor
img_resize = tensor_trans(img_resize)
print(img_resize.size)
writer.add_image("Resize_img", img_resize)

# Compose -> Resize -> ToTensor
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> Tensor
trans_compose = transforms.Compose([trans_resize_2, tensor_trans])
img_resize_2 = trans_compose(img)
writer.add_image("Resize_img_2", img_resize_2)

# RandomCrop使用
trans_crop = transforms.RandomCrop(256)
trans_compose_2 = transforms.Compose([trans_crop, tensor_trans])

for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop_img", img_crop, i)
    

writer.close()