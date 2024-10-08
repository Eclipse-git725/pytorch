from torch.utils.data import Dataset
from PIL import Image
import os

# help(Dataset)

# 继承dataset类
class MyData(Dataset):
    def __init__(self, root_dir, label_dir): 
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_list = os.listdir(self.path)
        

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_list)
    
# 实例化对象
root_dir = "dataset/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

img, label = ants_dataset[1]
# img.show()

train_data = ants_dataset + bees_dataset
print(len(train_data))

img, label = train_data[125]
img.show()