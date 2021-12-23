from torch.utils.tensorboard import SummaryWriter
from MyData import * 
import numpy as np
from torchvision import transforms

root_dir = "101_ObjectCategories"
ants_label_dir = "ant"
ants_dataset = MyData(root_dir,ants_label_dir)


writer = SummaryWriter("logs")
img2tensor = transforms.ToTensor()

for i in range(40):
    try:
        img ,label=ants_dataset[i]
        img_array = np.array(img)
        writer.add_image("test",img_array,i,dataformats='HWC')
        tensors_img = img2tensor(img)
    except:
        print(i)


for i in range(100):
    writer.add_scalar("y=x",i,i)


writer.close()