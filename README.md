# fcrn_pytorch
Deeper Depth Prediction with Fully Convolutional Residual Networks(2016 IEEE 3D Vision)的pytorch实现

论文：https://arxiv.org/pdf/1606.00373.pdf

主要参考：官方源码https://github.com/iro-cp/FCRN-DepthPrediction 
         前人实现https://github.com/XPFly1989/FCRN
>fcrn_pytorch: 文件结构
>>data:待处理的数据

>>>testIdxs.txt  trainIdxs.txt  nyu_depth_v2_labeled

>>model:保存模型

>>>NYU_ResNet-UpProj.npy model_300.pth

>>result:模型的效果

>>frcn.py:网络

>>loader.py:数据预处理

>>test.py:测试模型

>>train.py:可以继续训练模型

>>weights.py:加载官方给出的tensorflow参数

>>utils.py:功能函数

(1)下载NYU Depth Dataset V2 Labelled Dataset : http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat. 放在data文件夹中。备份地址：https://pan.baidu.com/s/1rIUbsEUjkZJheEZ5wTb5aA 密码: bfi4

(2)下载官方tensorflow的训练模型：http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy. 放在model文件夹中，也可以下载我训练的模型继续训练。

