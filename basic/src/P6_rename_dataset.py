import os

root_dir = "dataset/train"
# target_dir = "ants_image"
target_dir = "bees_image"
image_list = os.listdir(os.path.join(root_dir, target_dir))
label = target_dir.split("_")[0]
# out_dir = "ants_label"
out_dir = "bees_label"

for i in image_list:
    file_name = i.split(".jpg")[0]
    with open(os.path.join(root_dir, out_dir, "{}.txt".format(file_name)), "w") as f:
        f.write(label)
