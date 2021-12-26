import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=0)


    def forward(self,x):
        return self.conv1(x)

# test = torch.randint(0,10,size=(5,5))
# test = torch.randn(5,5)

# print(test)
# # test = torch.tensor([[1,2,1,1,0],
# #                         [2,3,1,1,0],
# #                         [2,3,1,1,0],
# #                         [2,3,1,1,0],
# #                         [1,2,1,0,0]])
# kernel = torch.randn(3,3)


# ken = nn.Conv2d(1,1,1,stride=1)



# test = torch.reshape(test,(1, 1, 5, 5))
# kernel = torch.reshape(kernel,(1, 1, 3, 3))

# print(ken(kernel).shape)
# print(test.shape)


# print(test)

# print(F.conv2d(test,kernel, stride=1))

# print(F.conv2d(test, kernel, stride=2, padding = 1))





