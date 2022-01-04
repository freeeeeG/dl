
import torch
from torch import nn


x = torch.tensor([1,2,3],dtype=torch.float32).reshape(1,1,1,3)
y = torch.tensor([1,2,5],dtype=torch.float32).reshape(1,1,1,3)



loss = nn.MSELoss()
res = loss(x,y)
print(res)
