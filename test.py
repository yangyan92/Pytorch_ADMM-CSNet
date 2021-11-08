"""
    ADMM_-CSNET test example (v1) with MR slices
    By Yan Yang, Jian Sun, Huibin Li, Zongben Xu

    Please cite the below paper for the code:

    Yan Yang, Jian Sun, Huibin Li, Zongben Xu. ADMM-CSNet: A Deep Learning Approach for Image Compressive Sensing,
    TPAMI(2019).
"""
from __future__ import print_function, division
import os
import argparse
from network.CSNet_Layers import CSNetADMMLayer
from utils.dataset import get_test_data
import torch.utils.data as data
from utils.my_loss import MyLoss
from utils.metric import complex_psnr
import gc
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
    parser.add_argument('--outf', type=str, default='logs_csnet', help='path of log files')
    args = parser.parse_args()

    ###############################################################################
    # load data info
    ###############################################################################
    test = get_test_data(args.data_dir)
    test_loader = data.DataLoader(dataset=test, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                  pin_memory=False)

    ###############################################################################
    # mask
    ###############################################################################
    dir = 'data/mask'
    data = loadmat(join(dir, os.listdir(dir)[0]))
    mask_data = data['mask']
    mask = ifftshift(torch.Tensor(mask_data)).cuda()

    ###############################################################################
    # Build model
    ###############################################################################
    print('Loading model ...\n')
    model = CSNetADMMLayer(mask).cuda()
    model.load_state_dict(torch.load(os.path.join(args.outf, 'cs_net_sample0.2.pth')))
    model.eval()

    ###############################################################################
    # loss
    ###############################################################################
    criterion = MyLoss().cuda()

    ###############################################################################
    # test
    ###############################################################################
    test_err = 0
    test_psnr = 0
    test_batches = 0
    for batch , (label,num) in enumerate(test_loader):
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