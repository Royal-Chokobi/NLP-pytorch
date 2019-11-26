import torch
import numpy as np

torch.manual_seed(1234)

def describe(x):
    print("Type : ", format(x.type()))
    print("Shape/size : ", format(x.shape))
    print("Values : \n", format(x))

'''
    torch.Tensor(행, 열) 
    A torch.Tensor는 단일 데이터 유형의 요소를 포함하는 다차원 매트릭스입니다.
    Torch는 9 개의 CPU 텐서 유형과 9 개의 GPU 텐서 유형을 정의합니다.
    torch.Tensor기본 텐서 유형 ( torch.FloatTensor) 의 별명입니다.
'''
describe(torch.Tensor(2, 5))
x = torch.BoolTensor(2, 5)
print(x)

z = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
print(z)

describe(torch.randn(2, 5))
y = torch.randn(2, 5)
print(y)

print("=======================================================\n\n")

ze = torch.zeros([2, 4], dtype=torch.int32)
print(ze)

cpu = torch.device('cpu')
oo = torch.FloatTensor(2,4).fill_(1)
print(oo)
print(oo.type())

o = torch.ones([2, 4], dtype=torch.float64, device=cpu)
print(o)
print(o.fill_(3))

x = torch.Tensor([[1, 2,],
                  [2, 4,]])
describe(x)

npy = np.random.rand(2, 3)
print(npy)
print(npy.dtype)
describe(torch.from_numpy(npy))
print(npy.dtype)


kk = torch.ones(())
kk.new_empty((2, 3))
print(kk)

x = torch.arange(6)
describe(x)

x = x.view(2, 3)
describe(x)
print("\n\ndim = 0 ==================================================================== ")
describe(torch.sum(x, dim=0)) # dim = 0 각 열을 중점으로 sum
print(" \n\ndim =1 ====================================================================")
describe(torch.sum(x, dim=1)) # dim = 1 각 행을 중점으로 sum

print("==================================================================== \n\n")
describe(torch.transpose(x, 0, 1))

print("==================================================================== \n\n")
print(x)
indices = torch.LongTensor([0, 2])
print(indices)
describe(torch.index_select(x, dim=1, index=indices))

indices = torch.LongTensor([1, 1])
describe(torch.index_select(x, dim=0, index=indices))

row_indices = torch.arange(2).long()
print(row_indices)
print(x)
col_indices = torch.LongTensor([0, 1])
describe(x[row_indices, col_indices])
describe(x[[0,1], [1,1]])


print("==================================================================== \n\n")
x = torch.arange(12).view(3, 4)
print(x)

x = x.unsqueeze(dim=1)
print(x)
print(torch.unsqueeze(x,dim =-1))

x = x.squeeze()
print(x)
print(x.reshape(4,3))
print(x.view(2,6))
print(x)

print("==================================================================== \n\n")

x = torch.arange(9).view(3,3)

print(x)
print("---")
new_x = torch.cat([x, x, x], dim=1)
print(new_x.shape)
print(new_x)


x = torch.arange(0, 12).view(3,4)
print("x: \n", x)
print("---")
print("x.tranpose(1, 0): \n", x.transpose(1, 0))


batch_size = 3
seq_size = 4
feature_size = 5

x = torch.arange(batch_size * seq_size * feature_size).view(batch_size, seq_size, feature_size)

print("x.shape: \n", x.shape)
print("x: \n", x)
print("-----")

print("x.permute(1, 0, 2).shape: \n", x.permute(1, 0, 2).shape)
print("x.permute(1, 0, 2): \n", x.permute(1, 0, 2))


print("==================================================================== \n\n")
x = torch.ones(3, 3, requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)

# a = torch.randn(2, 2)
# a = ((a * 3) / (a - 1))
# print(a.requires_grad)
# a.requires_grad_(True)
# print(a.requires_grad)
# b = (a * a).sum()
# print(b.grad_fn)

out.backward()
print(x.grad)
print(x)
