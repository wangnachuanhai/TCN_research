import torch
from train import train

x_train = torch.rand((128, 409, 6))
y_train = torch.randint(0, 16, (128, 1))

x_test = torch.rand((128, 409, 6))
y_test = torch.randint(0, 16, (128, 1))
print(torch.cuda.is_available())
pred_y_train, pred_y_test, loss_train = train(x_train, y_train, x_test, y_test, 1, 1, 0.01, 0.002, 0.3, 128, 'Adam')
print(pred_y_train, '\n', pred_y_test, '\n', loss_train)