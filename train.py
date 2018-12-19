
from loader import *
import os
from fcrn import FCRN
from torch.autograd import Variable
from weights import load_weights
from utils import load_split, loss_mse, loss_huber
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot

dtype = torch.cuda.FloatTensor
weights_file = "./model/NYU_ResNet-UpProj.npy"


def main():
    batch_size = 16
    data_path = './data/nyu_depth_v2_labeled.mat'
    learning_rate = 1.0e-4
    monentum = 0.9
    weight_decay = 0.0005
    num_epochs = 100


    # 1.Load data
    train_lists, val_lists, test_lists = load_split()
    print("Loading data...")
    train_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, train_lists),
                                               batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, val_lists),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, test_lists),
                                             batch_size=batch_size, shuffle=True, drop_last=True)
    print(train_loader)
    # 2.Load model
    print("Loading model...")
    model = FCRN(batch_size)
    model.load_state_dict(load_weights(model, weights_file, dtype)) #加载官方参数，从tensorflow转过来
    #加载训练模型
    resume_from_file = False
    resume_file = './model/model_300.pth'
    if resume_from_file:
        if os.path.isfile(resume_file):
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, checkpoint['epoch']))
        else:
            print("can not find!")
    model = model.cuda()

    # 3.Loss
    # 官方MSE
    # loss_fn = torch.nn.MSELoss()
    # 自定义MSE
    # loss_fn = loss_mse()
    # 论文的loss,the reverse Huber
    loss_fn = loss_huber()
    print("loss_fn set...")

    # 4.Optim
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("optimizer set...")

    # 5.Train
    best_val_err = 1.0e-4
    start_epoch = 0

    for epoch in range(num_epochs):
        print('Starting train epoch %d / %d' % (start_epoch + epoch + 1, num_epochs + start_epoch))
        model.train()
        running_loss = 0
        count = 0
        epoch_loss = 0
        for input, depth in train_loader:

            input_var = Variable(input.type(dtype))
            depth_var = Variable(depth.type(dtype))

            output = model(input_var)
            loss = loss_fn(output, depth_var)
            print('loss: %f' % loss.data.cpu().item())
            count += 1
            running_loss += loss.data.cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / count
        print('epoch loss:', epoch_loss)

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
                    plot.imsave('./result/gt_depth_epoch_{}.png'.format(start_epoch + epoch + 1), input_gt_depth_image, cmap="viridis")
                    plot.imsave('./result/pred_depth_epoch_{}.png'.format(start_epoch + epoch + 1), pred_depth_image, cmap="viridis")

                loss_local += loss_fn(output, depth_var)

                num_samples += 1

        err = float(loss_local) / num_samples
        print('val_error: %f' % err)

        if err < best_val_err or epoch == num_epochs - 1:
            best_val_err = err
            torch.save({
                'epoch': start_epoch + epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, './model/model_' + str(start_epoch + epoch + 1) + '.pth')

        if epoch % 10 == 0:
            learning_rate = learning_rate * 0.8


if __name__ == '__main__':
    main()