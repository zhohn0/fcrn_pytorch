import os
import torch
import numpy as np
from fcrn import FCRN
from train import load_split
from torch.autograd import Variable
from loader import NyuDepthLoader
import matplotlib.pyplot as plot

data_path = './data/nyu_depth_v2_labeled.mat'
dtype = torch.cuda.FloatTensor

batch_size = 1
resume_from_file = True
Threshold_1_25 = 0
Threshold_1_25_2 = 0
Threshold_1_25_3 = 0
RMSE_linear = 0.0
RMSE_log = 0.0
RMSE_log_scale_invariant = 0.0
ARD = 0.0
SRD = 0.0

model = FCRN(batch_size)
model = model.cuda()
loss_fn = torch.nn.MSELoss().cuda()

resume_file = './model/model_100.pth'

if resume_from_file:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_file))

_, _, test_lists = load_split()
num_samples = len(test_lists)

test_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, test_lists),
                                             batch_size=batch_size, shuffle=False, drop_last=False)
model.eval()
idx = 0
with torch.no_grad():
    for input, gt in test_loader:
        input_var = Variable(input.type(dtype))
        gt_var = Variable(gt.type(dtype))

        output = model(input_var)

        #input_rgb_image = input_var[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        input_rgb_image = input[0].data.permute(1, 2, 0)
        input_gt_depth_image = gt_var[0].data.squeeze().cpu().numpy().astype(np.float32)
        pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)

        input_gt_depth_image /= np.max(input_gt_depth_image)
        pred_depth_image /= np.max(pred_depth_image)

        idx = idx + 1
        if idx + 1 == len(test_loader):
            print('predict complete.')
            plot.imsave('Test_input_rgb_{:05d}.png'.format(idx), input_rgb_image)
            plot.imsave('Test_gt_depth_{:05d}.png'.format(idx), input_gt_depth_image, cmap="viridis")
            plot.imsave('Test_pred_depth_{:05d}.png'.format(idx), pred_depth_image, cmap="viridis")


        n = np.sum(input_gt_depth_image > 1e-3) #计算值大于1e-3的个数

        idxs = (input_gt_depth_image <= 1e-3) # 返回与原始数据同维的布尔值
        pred_depth_image[idxs] = 1 # 将小于1e-3赋值成1
        input_gt_depth_image[idxs] = 1

        pred_d_gt = pred_depth_image / input_gt_depth_image
        pred_d_gt[idxs] = 100
        gt_d_pred = input_gt_depth_image / pred_depth_image
        gt_d_pred[idxs] = 100

        Threshold_1_25 += np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25) / n #np.maximum返回相对较大的值
        Threshold_1_25_2 += np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25) / n
        Threshold_1_25_3 += np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25 * 1.25) / n

        log_pred = np.log(pred_depth_image)
        log_gt = np.log(input_gt_depth_image)

        d_i = log_gt - log_pred

        RMSE_linear += np.sqrt(np.sum((pred_depth_image - input_gt_depth_image) ** 2) / n)
        RMSE_log += np.sqrt(np.sum((log_pred - log_gt) ** 2) / n)
        RMSE_log_scale_invariant += np.sum(d_i ** 2) / n + (np.sum(d_i) ** 2) / (n ** 2)
        ARD += np.sum(np.abs((pred_depth_image - input_gt_depth_image)) / input_gt_depth_image) / n
        SRD += np.sum(((pred_depth_image - input_gt_depth_image) ** 2) / input_gt_depth_image) / n

Threshold_1_25 /= num_samples
Threshold_1_25_2 /= num_samples
Threshold_1_25_3 /= num_samples
RMSE_linear /= num_samples
RMSE_log /= num_samples
RMSE_log_scale_invariant /= num_samples
ARD /= num_samples
SRD /= num_samples

print('Threshold_1_25: {}'.format(Threshold_1_25))
print('Threshold_1_25_2: {}'.format(Threshold_1_25_2))
print('Threshold_1_25_3: {}'.format(Threshold_1_25_3))
print('RMSE_linear: {}'.format(RMSE_linear))
print('RMSE_log: {}'.format(RMSE_log))
print('RMSE_log_scale_invariant: {}'.format(RMSE_log_scale_invariant))
print('ARD: {}'.format(ARD))
print('SRD: {}'.format(SRD))

