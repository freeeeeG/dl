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
# torch.set_default_tensor_type(torch.IntTensor)
test = torch.randint(0,10,size=(5,5))
# test = torch.tensor([[1,2,1,1,0],
#                         [2,3,1,1,0],
#                         [2,3,1,1,0],
#                         [2,3,1,1,0],
#                         [1,2,1,0,0]])
kernel = torch.tensor([[1,2,1],
                        [2,3,1],
                        [1,2,1]])
test = torch.reshape(test,(1, 1, 5, 5))
kernel = torch.reshape(kernel,(1, 1, 3, 3))

print(test.dtype)
print(kernel.dtype)

print(F.conv2d(test, kernel, stride=1))

print(F.conv2d(test, kernel, stride=2, padding = 1))





