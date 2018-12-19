import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
# 自定义损失函数
class loss_huber(nn.Module):
    def __init__(self):
        super(loss_huber,self).__init__()

    def forward(self, pred, truth):
        c = pred.shape[1] #通道
        h = pred.shape[2] #高
        w = pred.shape[3] #宽
        pred = pred.view(-1, c * h * w)
        truth = truth.view(-1, c * h * w)
        # 根据当前batch所有像素计算阈值
        t = 0.2 * torch.max(torch.abs(pred - truth))
        # 计算L1范数
        l1 = torch.mean(torch.mean(torch.abs(pred - truth), 1), 0)
        # 计算论文中的L2
        l2 = torch.mean(torch.mean(((pred - truth)**2 + t**2) / t / 2, 1), 0)

        if l1 > t:
            return l2
        else:
            return l1

class loss_mse(nn.Module):
    def __init__(self):
        super(loss_mse, self).__init__()
    def forward(self, pred, truth):
        c = pred.shape[1]
        h = pred.shape[2]
        w = pred.shape[3]
        pred = pred.view(-1, c * h * w)
        truth = truth.view(-1, c * h * w)
        return torch.mean(torch.mean((pred - truth), 1)**2, 0)

if __name__ == '__main__':
    loss = loss_huber()
    x = torch.zeros(2, 1, 2, 2)
    y = torch.ones(2, 1, 2, 2)
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    r = loss(x, y)
    print(r)


# 加载数据集的index
def load_split():
    current_directoty = os.getcwd()
    train_lists_path = current_directoty + '/data/trainIdxs.txt'
    test_lists_path = current_directoty + '/data/testIdxs.txt'

    train_f = open(train_lists_path)
    test_f = open(test_lists_path)

    train_lists = []
    test_lists = []

    train_lists_line = train_f.readline()
    while train_lists_line:
        train_lists.append(int(train_lists_line) - 1)
        train_lists_line = train_f.readline()
    train_f.close()

    test_lists_line = test_f.readline()
    while test_lists_line:
        test_lists.append(int(test_lists_line) - 1)
        test_lists_line = test_f.readline()
    test_f.close()

    val_start_idx = int(len(train_lists) * 0.8)

    val_lists = train_lists[val_start_idx:-1]
    train_lists = train_lists[0:val_start_idx]

    return train_lists, val_lists, test_lists

# 测试网络
def validate(model, val_loader, loss_fn, dtype):
    # validate
    model.eval()
    num_correct, num_samples = 0, 0
    loss_local = 0
    with torch.no_grad():
        for input, depth in val_loader:
            input_var = Variable(input.type(dtype))
            depth_var = Variable(depth.type(dtype))

            output = model(input_var)
            if num_epochs == epoch + 1:
                # 关于保存的测试图片可以参考 loader 的写法
                # input_rgb_image = input_var[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                input_rgb_image = input[0].data.permute(1, 2, 0)
                input_gt_depth_image = depth_var[0][0].data.cpu().numpy().astype(np.float32)
                pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)

                input_gt_depth_image /= np.max(input_gt_depth_image)
                pred_depth_image /= np.max(pred_depth_image)

                plot.imsave('./result/input_rgb_epoch_{}.png'.format(start_epoch + epoch + 1), input_rgb_image)
                plot.imsave('./result/gt_depth_epoch_{}.png'.format(start_epoch + epoch + 1), input_gt_depth_image,
                            cmap="viridis")
                plot.imsave('./result/pred_depth_epoch_{}.png'.format(start_epoch + epoch + 1), pred_depth_image,
                            cmap="viridis")

            loss_local += loss_fn(output, depth_var)

            num_samples += 1

    err = float(loss_local) / num_samples
    print('val_error: %f' % err)