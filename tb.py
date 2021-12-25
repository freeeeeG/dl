from numpy.core.fromnumeric import resize
from numpy.core.numeric import False_
from torch.utils.data import dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms.transforms import Normalize
from MyData import * 
import numpy as np
from torchvision import transforms
import ssl
# 取消全局证书认证
ssl._create_default_https_context = ssl._create_unverified_context

root_dir = "101_ObjectCategories"
ants_label_dir = "ant"
ants_dataset = MyData(root_dir,ants_label_dir)


writer = SummaryWriter("logs")
img2tensor = transforms.ToTensor()
# 输入    PIL     Image.open()
# 输出    tensor      ToTensor()7
for i in range(40):
    try:
        img ,label=ants_dataset[i]
        img_array = np.array(img)
        writer.add_image("test",img_array,i,dataformats='HWC')
        # Normalize(归一化)
        tensors_img = img2tensor(img)
        print(tensors_img[0][0][0])
        trans_norm = transforms.Normalize([6, 3, 2],[9, 3, 5])
        img_norm = trans_norm(tensors_img)
        print(img_norm[0][0][0])
        writer.add_image("Normalize",img_norm,2)
        # Resize 调整大小
        trans_resize = transforms.Resize((512,512))
        img_resize = trans_resize(img)
        img_resize = img2tensor(img_resize)

        # Comepose - resize
        trans_resize_2 = transforms.Resize(512)
        trans_compose = transforms.Compose([trans_resize_2,img2tensor])
        img_resize_2 = trans_compose(img)
        # Randomcrop
        trans_random = transforms.RandomCrop(512)
        trans_compose2 = transforms.Compose([trans_random,img2tensor])
        for i in range(10):
            img_corp = trans_compose2(img)
            writer.add_image("corp",img_corp,i)
    except:
        print(i)


for i in range(100):
    writer.add_scalar("y=x",i,i)



dataset_transform = transforms.Compose([torchvision.transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root="./dataset", transform = dataset_transform, train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", transform = dataset_transform, train=False, download=True)

writer = SummaryWriter("test")
for i in range(10):
    img,label = test_set[i]
    writer.add_image("dataset", img, i)
writer.close()


writer.close()