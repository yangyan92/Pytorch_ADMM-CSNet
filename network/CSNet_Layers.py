import numpy as np
import torch.nn as nn
import torchpwl
from scipy.io import loadmat
from os.path import join
import os
from utils.fftc import *
import torch


class CSNetADMMLayer(nn.Module):
    def __init__(
        self,
        mask,
        in_channels: int = 1,
        out_channels: int = 128,
        kernel_size: int = 5

    ):
        """
        Args:

        """
        super(CSNetADMMLayer, self).__init__()

        self.rho = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.mask = mask
        self.re_org_layer = ReconstructionOriginalLayer(self.rho, self.mask)
        self.conv1_layer = ConvolutionLayer1(in_channels, out_channels, kernel_size)
        self.nonlinear_layer = NonlinearLayer()
        self.conv2_layer = ConvolutionLayer2(out_channels, in_channels, kernel_size)
        self.min_layer = MinusLayer()
        self.multiple_org_layer = MultipleOriginalLayer(self.gamma)
        self.re_update_layer = ReconstructionUpdateLayer(self.rho, self.mask)
        self.add_layer = AdditionalLayer()
        self.multiple_update_layer = MultipleUpdateLayer(self.gamma)
        self.re_final_layer = ReconstructionFinalLayer(self.rho, self.mask)
        layers = []

        layers.append(self.re_org_layer)
        layers.append(self.conv1_layer)
        layers.append(self.nonlinear_layer)
        layers.append(self.conv2_layer)
        layers.append(self.min_layer)
        layers.append(self.multiple_org_layer)

        for i in range(8):
            layers.append(self.re_update_layer)
            layers.append(self.add_layer)
            layers.append(self.conv1_layer)
            layers.append(self.nonlinear_layer)
            layers.append(self.conv2_layer)
            layers.append(self.min_layer)
            layers.append(self.multiple_update_layer)

        layers.append(self.re_update_layer)
        layers.append(self.add_layer)
        layers.append(self.conv1_layer)
        layers.append(self.nonlinear_layer)
        layers.append(self.conv2_layer)
        layers.append(self.min_layer)
        layers.append(self.multiple_update_layer)

        layers.append(self.re_final_layer)

        self.cs_net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1_layer.conv.weight = torch.nn.init.normal_(self.conv1_layer.conv.weight, mean=0, std=1)
        self.conv2_layer.conv.weight = torch.nn.init.normal_(self.conv2_layer.conv.weight, mean=0, std=1)
        self.conv1_layer.conv.weight.data = self.conv1_layer.conv.weight.data * 0.025
        self.conv2_layer.conv.weight.data = self.conv2_layer.conv.weight.data * 0.025

    def forward(self, x):
        y = torch.mul(x, self.mask)
        x = self.cs_net(y)
        x = torch.fft.ifft2(y+(1-self.mask)*torch.fft.fft2(x))
        return x


# reconstruction original layers
class ReconstructionOriginalLayer(nn.Module):
    def __init__(self, rho, mask):
        super(ReconstructionOriginalLayer,self).__init__()
        self.rho = rho
        self.mask = mask

    def forward(self, x):
        mask = self.mask
        denom = torch.add(mask.cuda(), self.rho)
        a = 1e-6
        value = torch.full(denom.size(), a).cuda()
        denom = torch.where(denom == 0, value, denom)
        orig_output1 = torch.div(1, denom)

        orig_output2 = torch.mul(x, orig_output1)
        orig_output3 = torch.fft.ifft2(orig_output2)
        # define data dict
        cs_data = dict()
        cs_data['input'] = x
        cs_data['conv1_input'] = orig_output3
        return cs_data


# reconstruction middle layers
class ReconstructionUpdateLayer(nn.Module):
    def __init__(self, rho, mask):
        super(ReconstructionUpdateLayer,self).__init__()
        self.rho = rho
        self.mask = mask

    def forward(self, x):
        minus_output = x['minus_output']
        multiple_output = x['multi_output']
        input = x['input']
        mask = self.mask
        number = torch.add(input, self.rho * torch.fft.fft2(torch.sub(minus_output, multiple_output)))
        denom = torch.add(mask.cuda(), self.rho)
        a = 1e-6
        value = torch.full(denom.size(), a).cuda()
        denom = torch.where(denom == 0, value, denom)
        orig_output1 = torch.div(1, denom)
        orig_output2 = torch.mul(number, orig_output1)
        orig_output3 = torch.fft.ifft2(orig_output2)
        x['re_mid_output'] = orig_output3
        return x


# reconstruction middle layers
class ReconstructionFinalLayer(nn.Module):
    def __init__(self, rho, mask):
        super(ReconstructionFinalLayer, self).__init__()
        self.rho = rho
        self.mask = mask

    def forward(self, x):
        minus_output = x['minus_output']
        multiple_output = x['multi_output']
        input = x['input']
        mask = self.mask
        number = torch.add(input, self.rho * torch.fft.fft2(torch.sub(minus_output, multiple_output)))
        denom = torch.add(mask.cuda(), self.rho)
        a = 1e-6
        value = torch.full(denom.size(), a).cuda()
        denom = torch.where(denom == 0, value, denom)
        orig_output1 = torch.div(1, denom)
        orig_output2 = torch.mul(number, orig_output1)
        orig_output3 = torch.fft.ifft2(orig_output2)
        x['re_final_output'] = orig_output3
        return x['re_final_output']


# multiple original layer
class MultipleOriginalLayer(nn.Module):
    def __init__(self,gamma):
        super(MultipleOriginalLayer,self).__init__()
        self.gamma = gamma

    def forward(self,x):
        org_output = x['conv1_input']
        minus_output = x['minus_output']
        output= torch.mul(self.gamma,torch.sub(org_output, minus_output))
        x['multi_output'] = output
        return x


# multiple middle layer
class MultipleUpdateLayer(nn.Module):
    def __init__(self,gamma):
        super(MultipleUpdateLayer,self).__init__()
        self.gamma = gamma

    def forward(self, x):
        multiple_output = x['multi_output']
        re_mid_output = x['re_mid_output']
        minus_output = x['minus_output']
        output= torch.add(multiple_output,torch.mul(self.gamma,torch.sub(re_mid_output , minus_output)))
        x['multi_output'] = output
        return x


# convolution layer
class ConvolutionLayer1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,kernel_size:int):
        super(ConvolutionLayer1,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int((kernel_size-1)/2), stride=1, dilation= 1,bias=True)

    def forward(self, x):
        conv1_input = x['conv1_input']
        real = self.conv(conv1_input.real)
        imag = self.conv(conv1_input.imag)
        output = torch.complex(real, imag)
        x['conv1_output'] = output
        return x


# convolution layer
class ConvolutionLayer2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ConvolutionLayer2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2),
                              stride=1, dilation=1, bias=True)

    def forward(self, x):
        nonlinear_output = x['nonlinear_output']
        real = self.conv(nonlinear_output.real)
        imag = self.conv(nonlinear_output.imag)
        output = torch.complex(real, imag)

        x['conv2_output'] = output
        return x


# nonlinear layer
class NonlinearLayer(nn.Module):
    def __init__(self):
        super(NonlinearLayer,self).__init__()
        self.pwl = torchpwl.PWL(num_channels=128, num_breakpoints=101)

    def forward(self, x):
        conv1_output = x['conv1_output']
        y_real = self.pwl(conv1_output.real)
        y_imag = self.pwl(conv1_output.imag)
        output = torch.complex(y_real, y_imag)
        x['nonlinear_output'] = output
        return x


# minus layer
class MinusLayer(nn.Module):
    def __init__(self):
        super(MinusLayer, self).__init__()

    def forward(self, x):
        minus_input = x['conv1_input']
        conv2_output = x['conv2_output']
        output= torch.sub(minus_input, conv2_output)
        x['minus_output'] = output
        return x


# addtional layer
class AdditionalLayer(nn.Module):
    def __init__(self):
        super(AdditionalLayer,self).__init__()

    def forward(self, x):
        mid_output = x['re_mid_output']
        multi_output = x['multi_output']
        output= torch.add(mid_output,multi_output)
        x['conv1_input'] = output
        return x