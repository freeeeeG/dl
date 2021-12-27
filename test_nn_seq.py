
from os import write
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter



# class Linear_test(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3,32,5,padding=2)
#         self.max_pool1 = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(32,32,5,padding=2)
#         self.max_pool2 = nn.MaxPool2d(2)
#         self.conv3 = nn.Conv2d(32,64,5,padding=2)
#         self.max_pool3 = nn.MaxPool2d(2)
#         self.fla = nn.Flatten()
#         self.linear1 = nn.Linear(1024,64)
#         self.linear2 = nn.Linear(64,10)


#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.max_pool1(x)
#         x = self.conv2(x)
#         x = self.max_pool2(x)
#         x = self.conv3(x)
#         x = self.max_pool3(x)

#         x = self.fla(x)
#         x = self.linear1(x)
#         x = self.linear2(x)
#         return x




class Linear_test(nn.Module):
    def __init__(self):
        super().__init__()
        self.module1 = nn.Sequential(
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
        x = self.module1(x)
        return x
test = Linear_test()
print(test)
input = torch.ones((64,3,32,32))
output = test(input)
print(output.shape)

writer = SummaryWriter("logs_seq")
writer.add_graph(test,input)
writer.close()
