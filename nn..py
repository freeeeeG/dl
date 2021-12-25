import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,20,5)
        self.conv2 = nn.Conv2d(1,20,5)

    def forward(self,x):
        F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
        return x+1
# torch.set_default_tensor_type(torch.DoubleTensor)
test = torch.rand(5,5)
# test = torch.tensor([[1,2,1,1,0],
#                         [2,3,1,1,0],
#                         [2,3,1,1,0],
#                         [2,3,1,1,0],
#                         [1,2,1,0,0]])
kernel = torch.tensor([[1,2,1],
                        [2,3,1],
                        [1,2,1]])
# print(kernel.dtype)
test = torch.reshape(test,(1, 1, 5, 5))
kernel = torch.reshape(kernel,(1, 1, 3, 3))

print(out = F.conv2d(test,kernel,stride=1))
