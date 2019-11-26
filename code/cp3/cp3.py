import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x = torch.arange(-5., 5., 0.1)
print(x)
y = torch.sigmoid(x)
plt.plot(x.numpy(), y.detach().numpy())
plt.show()

y1 = torch.tanh(x)

plt.plot(x.numpy(), y1.numpy())
plt.show()

relu = torch.nn.ReLU()
y2 = relu(x)
plt.plot(x.numpy(), y2.numpy())
plt.show()

prelu = torch.nn.PReLU(num_parameters=1)
y3 = prelu(x)
plt.plot(x.numpy(), y3.detach().numpy())
plt.show()

softmax = torch.nn.Softmax(dim=1)
x_input = torch.randn(1, 30)
y_output = softmax(x_input)

print(x_input)
print(y_output)
print(torch.sum(y_output, dim=1))


mse_loss = nn.MSELoss()
outputs = torch.randn(3, 5, requires_grad=True)
print(outputs)
targets = torch.randn(3, 5)
print(targets)
loss = mse_loss(outputs, targets)
loss.backward()
print(loss)