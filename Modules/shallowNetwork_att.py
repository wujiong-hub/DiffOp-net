import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import numbers
from .network_others import *

class CoordAtt(nn.Module):
    def __init__(self, oup):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_d = nn.AdaptiveAvgPool3d((1, 1, None))

        self.conv_h = nn.Conv3d(oup, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(oup, oup, kernel_size=1, stride=1, padding=0)
        self.conv_d = nn.Conv3d(oup, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, copy, up):
        identity = copy
        copy_h = self.pool_h(copy)
        copy_w = self.pool_w(copy)
        copy_d = self.pool_d(copy)
        
        up_h = self.pool_h(up)
        up_w = self.pool_w(up)
        up_d = self.pool_d(up)

        copyup_h = self.conv_h(up_d*up_w*copy_h).sigmoid() 
        copyup_w = self.conv_w(up_d*copy_w*up_h).sigmoid()
        copyup_d = self.conv_d(copy_d*up_w*up_h).sigmoid()

        out = identity * copyup_d * copyup_w * copyup_h

        return out


class MDRNet_r1(nn.Module):
    def __init__(self, in_channel=2, n_dim=3, nv=1, start_channel=16):
        self.in_channel = in_channel  #inchannel=2
        self.n_dim = n_dim
        self.start_channel = start_channel #start_channel=16
        self.nv = nv

        bias_opt = True
 
        super(MDRNet_r1, self).__init__()
        self.c0 = self.encoder(self.in_channel, self.start_channel, stride=1, bias=bias_opt)  #2->16
        self.c1 = self.encoder(self.start_channel, self.start_channel*2, stride=2, bias=bias_opt)  #16->32
        self.c2 = self.encoder(self.start_channel*2, self.start_channel * 2, stride=2, bias=bias_opt)  #32->32
        self.c3 = self.encoder(self.start_channel * 2, self.start_channel * 2, stride=2, bias=bias_opt) #32->32
        self.c4 = self.encoder(self.start_channel * 2, self.start_channel * 2, stride=2, bias=bias_opt) #32->32

        self.d1 = self.decoder(self.start_channel * 2, self.start_channel * 2, stride=2, bias=bias_opt) 

        self.d2 = self.decoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 2,  stride=2, bias=bias_opt) #32+32->32
        self.d3 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 2,  stride=1, bias=bias_opt) #32+32->32

        self.d4 = self.encoder(self.start_channel * 2, self.start_channel, stride=1, bias=bias_opt)   #32->16

        self.out = nn.ModuleList([self.outputs(self.start_channel, n_dim, stride=1, bias=bias_opt) for i in range(self.nv)])

        self.CoordAtt1 = CoordAtt(self.start_channel * 2)
        self.CoordAtt2 = CoordAtt(self.start_channel * 2)


    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.LeakyReLU(0.2))
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            )
        return layer

    def forward(self, x,y):
        x_in = torch.cat((x, y), 1)
        e0 = self.c0(x_in)
        e1 = self.c1(e0)
        e2 = self.c2(e1)
        e3 = self.c3(e2)
        e4 = self.c4(e3)
        d1 = self.d1(e4)
        d2 = torch.cat((d1, self.CoordAtt1(e3, d1)), 1)
        d2 = self.d2(d2)
        d3 = torch.cat((d2, self.CoordAtt2(e2, d2)), 1)
        d3 = self.d3(d3)
        d4 = self.d4(d3)

        size = x.shape
        tvs = torch.zeros((self.nv, self.n_dim, size[2]//4, size[3]//4, size[4]//4)).cuda()
        for i, l in enumerate(self.out):
            tvs[i,:,:,:,:] = l(d4)[0,:,:,:,:]
        return tvs


class MDRNet_r2(nn.Module):
    def __init__(self, in_channel=2, n_dim=3, nv=1, start_channel=16):
        self.in_channel = in_channel  #inchannel=2
        self.n_dim = n_dim
        self.start_channel = start_channel #start_channel=16
        self.nv = nv
        bias_opt = True
 
        super(MDRNet_r2, self).__init__()
        self.c0 = self.encoder(self.in_channel, self.start_channel, stride=1, bias=bias_opt)  #2->16
        self.c1 = self.encoder(self.start_channel, self.start_channel*2, stride=2, bias=bias_opt)  #16->32
        self.c2 = self.encoder(self.start_channel*2, self.start_channel * 2, stride=2, bias=bias_opt)  #32->32
        self.c3 = self.encoder(self.start_channel * 2, self.start_channel * 2, stride=2, bias=bias_opt) #32->32
        self.c4 = self.encoder(self.start_channel * 2, self.start_channel * 2, stride=2, bias=bias_opt) #32->32

        self.d1 = self.decoder(self.start_channel * 2, self.start_channel * 2, stride=2, bias=bias_opt) 

        self.d2 = self.decoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 2,  stride=2, bias=bias_opt) #32+32->32
        self.d3 = self.decoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 2,  stride=2, bias=bias_opt) #32+32->32

        self.d4 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 2,  stride=1, bias=bias_opt) #32+32->32
        self.d5 = self.encoder(self.start_channel * 2, self.start_channel, stride=1, bias=bias_opt)   #32->16

        self.out = nn.ModuleList([self.outputs(self.start_channel, n_dim, stride=1, bias=bias_opt) for i in range(self.nv)])

        self.CoordAtt1 = CoordAtt(self.start_channel * 2)
        self.CoordAtt2 = CoordAtt(self.start_channel * 2)
        self.CoordAtt3 = CoordAtt(self.start_channel * 2)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.LeakyReLU(0.2))
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            )
        return layer

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        e0 = self.c0(x_in)
        e1 = self.c1(e0)
        e2 = self.c2(e1)
        e3 = self.c3(e2)
        e4 = self.c4(e3)
        d1 = self.d1(e4)
        d2 = torch.cat((d1, self.CoordAtt1(e3, d1)), 1)
        d2 = self.d2(d2)
        d3 = torch.cat((d2, self.CoordAtt2(e2, d2)), 1)
        d3 = self.d3(d3)
        d4 = torch.cat((d3, self.CoordAtt3(e1, d3)), 1)
        d4 = self.d4(d4)

        d5 = self.d5(d4)

        size = x.shape
        tvs = torch.zeros((self.nv, self.n_dim, size[2]//2, size[3]//2, size[4]//2)).cuda()
        for i, l in enumerate(self.out):
            tvs[i,:,:,:,:] = l(d5)[0,:,:,:,:]
        return tvs


class MDRNet_r3(nn.Module):
    def __init__(self, in_channel=2, n_dim=3, nv=1, start_channel=16):
        self.in_channel = in_channel  #inchannel=2
        self.n_dim = n_dim
        self.start_channel = start_channel #start_channel=16
        self.nv = nv

        bias_opt = True

        super(MDRNet_r3, self).__init__()
        self.c0 = self.encoder(self.in_channel, self.start_channel, stride=1, bias=bias_opt)  #2->16
        self.c1 = self.encoder(self.start_channel, self.start_channel*2, stride=2, bias=bias_opt)  #16->32
        self.c2 = self.encoder(self.start_channel*2, self.start_channel * 2, stride=2, bias=bias_opt)  #32->32
        self.c3 = self.encoder(self.start_channel * 2, self.start_channel * 2, stride=2, bias=bias_opt)  #32->32
        self.c4 = self.encoder(self.start_channel * 2, self.start_channel * 2, stride=2, bias=bias_opt)  #32->32

        self.d1 = self.decoder(self.start_channel * 2, self.start_channel * 2, stride=2, bias=bias_opt) 

        self.d2 = self.decoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 2,  stride=2, bias=bias_opt) #32+32->32
        self.d3 = self.decoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 2,  stride=2, bias=bias_opt) #32+32->32
        self.d4 = self.decoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 2,  stride=2, bias=bias_opt) #32+32->32

        self.d5 = self.encoder(self.start_channel * 2, self.start_channel, stride=1, bias=bias_opt)   #32->16

        self.out = nn.ModuleList([self.outputs(self.start_channel, n_dim, stride=1, bias=bias_opt) for i in range(self.nv)])

        self.CoordAtt1 = CoordAtt(self.start_channel * 2)
        self.CoordAtt2 = CoordAtt(self.start_channel * 2)
        self.CoordAtt3 = CoordAtt(self.start_channel * 2)


    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.LeakyReLU(0.2))
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
            )
        return layer

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        e0 = self.c0(x_in)
        e1 = self.c1(e0)
        e2 = self.c2(e1)
        e3 = self.c3(e2)
        e4 = self.c4(e3)
        d1 = self.d1(e4)
        d2 = torch.cat((d1, self.CoordAtt1(e3, d1)), 1)
        d2 = self.d2(d2)
        d3 = torch.cat((d2, self.CoordAtt2(e2, d2)), 1)
        d3 = self.d3(d3)
        d4 = torch.cat((d3, self.CoordAtt3(e1, d3)), 1)
        d4 = self.d4(d4)

        d5 = self.d5(d4)

        size = x.shape
        tvs = torch.zeros((self.nv, self.n_dim, size[2], size[3], size[4])).cuda()
        for i, l in enumerate(self.out):
            tvs[i,:,:,:,:] = l(d5)[0,:,:,:,:]
        return tvs