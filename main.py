"""
    ADMM_-CSNET training example (v1) with MR slices
    By Yan Yang, Jian Sun, Huibin Li, Zongben Xu

    Please cite the below paper for the code:

    Yan Yang, Jian Sun, Huibin Li, Zongben Xu. ADMM-CSNet: A Deep Learning Approach for Image Compressive Sensing,
    TPAMI(2019).
"""
from __future__ import print_function, division
import sys
import os
import torch
import argparse
from network.CSNet_Layers import CSNetADMMLayer
from utils.dataset import get_data
import torch.utils.data as data
from utils.my_loss import MyLoss
import time
from utils.metric import complex_psnr
from tensorboardX import SummaryWriter
import gc
import torchvision.utils as utils
from scipy.io import loadmat
from utils.fftc import *
from os.path import join

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    ###############################################################################
    # parameters
    ###############################################################################
    parser = argparse.ArgumentParser(description=' main ')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='directory of data')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_epoch', default=1000, type=int, help='number of epochs')
    parser.add_argument('--save_dir', default='save_dir',
                        type=str,
                        help='directory to save results')
    parser.add_argument('--outf', type=str, default='logs_csnet', help='path of log files')
    args = parser.parse_args()

    ###############################################################################
    # callable methods
    ###############################################################################
    def try_make_dir(d):
        if not os.path.isdir(d):
            os.mkdir(d)

    def adjust_learning_rate(opt, epo, lr):
        """Sets the learning rate to the initial LR decayed by 5 every 50 epochs"""
        lr = lr * (0.5 ** (epo // 50))
        for param_group in opt.param_groups:
            param_group['lr'] = lr


    ###############################################################################
    # dataset
    ###############################################################################
    train, test, validate = get_data(args.data_dir)
    len_train, len_test, len_validate = len(train), len(test), len(validate)
    print("len_train: ", len_train, "\tlen_test:", len_test, "\tlen_test:", len_test)
    train_loader = data.DataLoader(dataset=train, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                   pin_memory=False)
    test_loader = data.DataLoader(dataset=test, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                  pin_memory=False)
    valid_loader = data.DataLoader(dataset=validate, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                   pin_memory=False)

    ###############################################################################
    # mask
    ###############################################################################
    dir = 'data/mask'
    data = loadmat(join(dir, os.listdir(dir)[0]))
    mask_data = data['mask']
    mask = ifftshift(torch.Tensor(mask_data)).cuda()

    ###############################################################################
    # ADMM-CSNET model
    ###############################################################################
    model = CSNetADMMLayer(mask).cuda()

    ###############################################################################
    # Adam optimizer
    ###############################################################################
    optimizer = torch.optim.Adam(model.parameters())

    ###############################################################################
    # self-define loss
    ###############################################################################
    criterion = MyLoss().cuda()

    writer = SummaryWriter(args.outf)
    ###############################################################################
    # train
    ###############################################################################
    print("start training...")
    start_time = time.time()
    for epoch in range(0, args.num_epoch + 1):
        total_loss_org = 0
        train_batches = 0
        train_psnr = 0
        adjust_learning_rate(optimizer, epoch, lr=0.002)
        # ===================train==========================
        for batch_idx, (label, num) in enumerate(train_loader):
            full_kspace = torch.fft.fft2(label.cuda())
            output = model(full_kspace)
            optimizer.zero_grad()
            loss_normal = criterion(output, label.cuda())
            loss_normal.backward()
            optimizer.step()
            total_loss_org += loss_normal.data.item()
            train_batches += 1
            train_psnr_value = complex_psnr(abs(output).cpu().detach().numpy(), abs(label).cpu().detach().numpy(),
                                            peak='normalized')
            train_psnr += train_psnr_value
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                  (epoch + 1, batch_idx + 1, len(train_loader), total_loss_org / (batch_idx + 1),
                   train_psnr / (batch_idx + 1)))
        train_psnr /= train_batches
        total_loss_org /= train_batches
        print("train_loss: ", total_loss_org)
        print("train_psnr: ", train_psnr)
        writer.add_scalar('psnr on train data', train_psnr, epoch)
        if epoch % 10 == 0:
            try_make_dir(args.save_dir + '/end_to_end_model/')
            torch.save(model.state_dict(),
                       os.path.join(args.outf, 'model{}.pth'.format(epoch)))
        model.eval()
        ###############################################################################
        # validate
        ###############################################################################
        validate_err = 0
        validate_psnr = 0
        validate_batches = 0
        with torch.no_grad():
            for batch_idx, (label, num) in enumerate(valid_loader):
                gc.collect()
                torch.cuda.empty_cache()
                full_kspace = torch.fft.fft2(label.cuda())
                val_output = model(full_kspace)
                validate_loss_normal = criterion(val_output, label.cuda())
                validate_err += validate_loss_normal.item()
                validate_batches += 1
                valid_psnr_value = complex_psnr(abs(val_output).cpu().numpy(), abs(label).cpu().numpy(),
                                                peak='normalized')
                validate_psnr += valid_psnr_value
                if epoch % 10 == 0:
                    resconstructed_image = utils.make_grid(abs(val_output.data.squeeze().cpu()), nrow=5, normalize=True,
                                                           scale_each=True)
                    writer.add_image('reconstructed image', resconstructed_image, epoch)

        validate_err /= validate_batches
        validate_psnr /= validate_batches
        print("valid_loss ", validate_err)
        print("valid_psnr ", validate_psnr)
        writer.add_scalar('psnr on valid data', validate_psnr, epoch)
        ###############################################################################
        # test
        ###############################################################################
        test_err = 0
        test_psnr = 0
        test_batches = 0
        model.eval()
        for batch_idx, (label, num) in enumerate(test_loader):
            gc.collect()
            with torch.no_grad():
                full_kspace = torch.fft.fft2(label.cuda())
                test_output = model(full_kspace)
                test_loss_normal = criterion(test_output, label.cuda())
                test_err += test_loss_normal.item()
                test_batches += 1
                test_psnr_value = complex_psnr(abs(test_output).cpu().numpy(), abs(label).cpu().numpy(),
                                               peak='normalized')
                test_psnr += test_psnr_value
        test_err /= test_batches
        test_psnr /= test_batches
        print("test_loss ", test_err)
        print("test_psnr ", test_psnr)
        writer.add_scalar('psnr on test data', test_psnr, epoch)
