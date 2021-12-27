

from _typeshed import Self
from numpy.core.numeric import outer
from torch import nn
from torch.nn.modules.linear import Linear


class Linear_test(nn.Module):
    def __init__(self,x,y):
        super().__init__()
        self.Linear1 = Linear(x,y)

    def forward(self,input):
        output =  self.Linear1(input)
        return output




