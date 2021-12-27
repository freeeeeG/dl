
import torch

from torch import nn

input = torch.reshape(torch.rand(5,5),(-1,1,5,5))


class Pool_Test(nn.Module):
    def __init__(self):
        super(Pool_Test,self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size= 3, ceil_mode= True)
    def forward(self, input):
        output = self.maxpool1(input)
        return output


