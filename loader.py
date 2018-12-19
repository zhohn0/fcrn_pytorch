import os
import numpy as np
import h5py
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import load_split


class NyuDepthLoader(data.Dataset):
    def __init__(self, data_path, lists):
        self.data_path = data_path
        self.lists = lists

        self.nyu = h5py.File(self.data_path)

        self.imgs = self.nyu['images']
        self.dpts = self.nyu['depths']

    def __getitem__(self, index):
        img_idx = self.lists[index]
        img = self.imgs[img_idx].transpose(2, 1, 0) # HWC
        dpt = self.dpts[img_idx].transpose(1, 0)
        img = Image.fromarray(img)
        dpt = Image.fromarray(dpt)
        input_transform = transforms.Compose([transforms.Resize(228),
                                              transforms.ToTensor()])

        target_depth_transform = transforms.Compose([transforms.Resize(228),
                                                     transforms.ToTensor()])

        img = input_transform(img)
        dpt = target_depth_transform(dpt)
        return img, dpt

    def __len__(self):
        return len(self.lists)

# 测试数据加载
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def test_loader():
    batch_size = 16
    data_path = './data/nyu_depth_v2_labeled.mat'
    # 1.Load data
    train_lists, val_lists, test_lists = load_split()

    print("Loading data...")
    train_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, train_lists),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    for input, depth in train_loader:
        print(input.size())
        break
    #input_rgb_image = input[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    input_rgb_image = input[0].data.permute(1, 2, 0)
    input_gt_depth_image = depth[0][0].data.cpu().numpy().astype(np.float32)

    input_gt_depth_image /= np.max(input_gt_depth_image)
    plt.imshow(input_rgb_image)
    plt.show()
    plt.imshow(input_gt_depth_image, cmap="viridis")
    plt.show()
    # plot.imsave('input_rgb_epoch_0.png', input_rgb_image)
    # plot.imsave('gt_depth_epoch_0.png', input_gt_depth_image, cmap="viridis")

if __name__ == '__main__':
    test_loader()

















