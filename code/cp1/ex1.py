import torch

def f(x):
    if (x.data > 0).all():
        return torch.sin(x)
    else:
        return torch.cos(x)

x = torch.tensor([1.0], requires_grad=True)
print(x)
y = f(x)
y.backward()
print(x.grad)


x = torch.tensor([1.0, 0.5], requires_grad=True)
y = f(x)
y.sum().backward()
print(x.grad)

def f2(x):
    mask = torch.gt(x, 0).float()
    return mask * torch.sin(x) + (1 - mask) * torch.cos(x)

x = torch.tensor([1.0, -1], requires_grad=True)
y = f2(x)
y.sum().backward()
print(x.grad)

def describe_grad(x):
    if x.grad is None:
        print("No gradient information")
    else:
        print("Gradient: \n{}".format(x.grad))
        print("Gradient Function: {}".format(x.grad_fn))


def describe(x):
    print("Type : ", format(x.type()))
    print("Shape/size : ", format(x.shape))
    print("Values : \n", format(x))


x = torch.ones(2, 2, requires_grad=True)
describe(x)
describe_grad(x)
print("--------")

y = (x + 2) * (x + 5) + 3
describe(y)
z = y.mean()
describe(z)
describe_grad(x)
print("--------")
z.backward(create_graph=True, retain_graph=True)
describe_grad(x)
print("--------")

x = torch.ones(2, 2, requires_grad=True)

y = x + 2
print(y.grad_fn)