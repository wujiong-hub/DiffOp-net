#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch
import numpy as np
import lddmm
import nibabel as ni
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


def applykernel(Vx, Vy, Vz, kernel):
    kernel = kernel.cuda()
    fftVx = torch.fft.fftn(Vx, norm="forward")
    ifftVx = torch.fft.ifftn(fftVx*kernel, norm="forward")
    fftVy = torch.fft.fftn(Vy, norm="forward")
    ifftVy = torch.fft.ifftn(fftVy*kernel, norm="forward")
    fftVz = torch.fft.fftn(Vz, norm="forward")
    ifftVz = torch.fft.ifftn(fftVz*kernel, norm="forward")
    del fftVx, fftVy, fftVz
    return ifftVx, ifftVy, ifftVz

def imageEnergy(MI, FI, Vsizex, Vsizey, Vsizez, scale, sigma, voxelVolum):
    MI = MI
    FI = FI
    value = lddmm.ssdMetric(MI, FI, Vsizex, Vsizey, Vsizez, scale)
    return  value.cpu()*voxelVolum*0.5/float(sigma)


def getForwardImage(tvs, templateImage, forg, morg, scale):
    timesVaryVelx = tvs[:,0,:,:,:]
    timesVaryVely = tvs[:,1,:,:,:]
    timesVaryVelz = tvs[:,2,:,:,:]
    
    phix, phiy, phiz = lddmm.intergrateVelocity(
                                    timesVaryVelx.contiguous(), 
                                    timesVaryVely.contiguous(), 
                                    timesVaryVelz.contiguous(), 
                                    1.0,
                                    forg[0],
                                    forg[1],
                                    forg[2],
                                    -1.0/float((timesVaryVelx.shape[0]+1)),
                                    int(timesVaryVelx.shape[0]-1),
                                    0.0,
                                    timesVaryVelx.shape[0]+1,
                                    scale)
  
    m_ForwardImage = lddmm.imageApplyField(
                                templateImage.contiguous(),
                                phix.contiguous(),
                                phiy.contiguous(),
                                phiz.contiguous(),
                                forg[0],
                                forg[1],
                                forg[2],
                                morg[0],
                                morg[1],
                                morg[2],
                                scale)
                                
    del phix, phiy, phiz, timesVaryVelx, timesVaryVely, timesVaryVelz 
    return m_ForwardImage


def getNablaEnergy(tvs, m_ForwardImage, targetImage, Akernel, forg, scale, sigma):
    NablaE = torch.zeros_like(tvs)
    timesVaryVelx = tvs[:,0,:,:,:]
    timesVaryVely = tvs[:,1,:,:,:]
    timesVaryVelz = tvs[:,2,:,:,:]
    
    for j in range(0,tvs.shape[0]): 
        
        if tvs.shape[0] == 1:
            phix = torch.zeros_like(timesVaryVelx[0,:,:,:])
            phiy = torch.zeros_like(timesVaryVelx[0,:,:,:])
            phiz = torch.zeros_like(timesVaryVelx[0,:,:,:])
        else:
            t = j/float(timesVaryVelx.shape[0]-1)

            if (j == timesVaryVelx.shape[0]-1):
                phix = torch.zeros_like(timesVaryVelx[0,:,:,:])
                phiy = torch.zeros_like(timesVaryVelx[0,:,:,:])
                phiz = torch.zeros_like(timesVaryVelx[0,:,:,:]) 
            else:
                phix, phiy, phiz = lddmm.intergrateVelocity(
                                    timesVaryVelx.contiguous(), 
                                    timesVaryVely.contiguous(), 
                                    timesVaryVelz.contiguous(), 
                                    t,
                                    forg[0],
                                    forg[1],
                                    forg[2],
                                    (1.0-t)/float((timesVaryVelx.shape[0]-1-j)+2),
                                    timesVaryVelx.shape[0]-1,
                                    1.0,
                                    timesVaryVelx.shape[0]+1-j,
                                    scale)

        Derivatx, Derivaty, Derivatz = lddmm.ccMetricDerivative(
                                        m_ForwardImage.contiguous(),
                                        targetImage.contiguous(),
                                        phix.contiguous(),
                                        phiy.contiguous(),
                                        phiz.contiguous(),
                                        scale,
                                        #1.0/float(sigma**2),
                                        1.0/float(sigma),
                                        4) 
        del phix, phiy, phiz
        Derivatx, Derivaty, Derivatz = applykernel(Derivatx.contiguous(), Derivaty.contiguous(), Derivatz.contiguous(), Akernel.contiguous())
        NablaEx = Derivatx.unsqueeze(0)
        NablaEy = Derivaty.unsqueeze(0)
        NablaEz = Derivatz.unsqueeze(0)
        NablaE[j,:,:,:,:] = torch.cat((NablaEx, NablaEy, NablaEz), dim=0)
        del NablaEx, NablaEy, NablaEz
            
    NablaE = tvs + NablaE 
    return NablaE



def getImageEnergy(tvs, forwardImage, targetImage, scale, sigma, voxelVolum):
    newImageEnergy = 0.0
    for iv in range(tvs.shape[0]):
        newImageEnergy += imageEnergy(forwardImage[iv,:,:,:], targetImage[iv,:,:,:], 
                                      tvs.shape[2],tvs.shape[3],tvs.shape[4],
                                      scale,sigma,voxelVolum)

    return newImageEnergy/tvs.shape[0]


class EnergyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tvs, forwardImage, targetImage, Lkernel, Akernel, forg, voxelVolum, scale, sigma):
        ctx.save_for_backward(tvs, forwardImage, targetImage, Akernel, forg, scale, sigma)
        newImageEnergy = imageEnergy(forwardImage, targetImage, tvs.shape[2], tvs.shape[3], tvs.shape[4], scale, sigma, voxelVolum)
        return newImageEnergy

    @staticmethod
    def backward(ctx, grad_out):
        tvs, forwardImage, targetImage, Akernel, forg, scale, sigma = ctx.saved_tensors
        NablaE = getNablaEnergy(tvs, forwardImage, targetImage, Akernel, forg, scale, sigma)
        del tvs, forwardImage, targetImage
        return NablaE, torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)


