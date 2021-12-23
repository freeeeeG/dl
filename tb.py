from torch.utils.tensorboard import SummaryWriter
from MyData import * 
import numpy as np


root_dir = "101_ObjectCategories"
ants_label_dir = "ant"
ants_dataset = MyData(root_dir,ants_label_dir)
img ,label=ants_dataset[0]
img_array = np.array(img)

writer = SummaryWriter("logs")


for i in range(40):
    try:
        writer.add_image("test",img_array,i,dataformats='HWC')
    except:
        print(i)


for i in range(100):
    writer.add_scalar("y=x",i,i)


writer.close()