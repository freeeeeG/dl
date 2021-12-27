
class A:
    def fun(self):
        print('A.fun')

class B(A):
    def fun(self):
        super().fun()
        print('B.fun')

class C(A):
    def fun(self):
        super().fun()
        print('C.fun')

class D(B , C):
    def fun(self):
        super().fun()
        print('D.fun')

D().fun()