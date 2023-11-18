import os
import torch

from torch import nn
from myModel import mytcn
import numpy as np

def train(x_train, y_train, x_test, y_test, gpu, EPOCH, LR, WD, DROP, BATCH, optimizername):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if torch.cuda.is_available():
        print('available')

    net = mytcn(6, [32, 64, 17], DROP, BATCH)
    model = net.cuda()
    print(model)

    if optimizername == 'SGD':
        print('Use SGD')
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WD, momentum=0.9)  # optimize all parameters
    else:
        print('Use Adam')
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
# 训练阶段
        print("Epoch:" + str(epoch))
        model.train()
        if net.modelname == 'tcn':
            x_train = torch.Tensor(np.transpose(np.array(x_train, dtype='float32'), (0, 2, 1)))
        y_train = y_train.type(torch.LongTensor)
        y_train = y_train.view(BATCH,)

        # 拟合数据得到模型输出
        output = model(x_train.cuda())
        pred_y_train = torch.max(output, 1)[1]

        loss_train = loss_func(output, y_train.cuda())
        optimizer.zero_grad()  # clear gradients for this training step
        loss_train.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients to update weights

# 测试阶段
        model.eval()
        with torch.no_grad():
            # 整理x, y成为网络需要的shape和type
            if net.modelname == 'tcn':
                x_test = torch.Tensor(np.transpose(np.array(x_test, dtype='float32'), (0, 2, 1)))
            y_test = y_test.type(torch.LongTensor)
            y_test = y_test.view(BATCH, )

            # 拟合数据得到模型输出
            output_test = model(x_test.cuda())
            pred_y_test = torch.max(output_test, 1)[1]

    return pred_y_train, pred_y_test, loss_train