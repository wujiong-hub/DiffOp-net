import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import os.path as osp
import nibabel as ni
import numpy as np
import lddmm
import numbers
import math
import SimpleITK as sitk
from .EnergyFunction import *

def getFieldEnd(tvs, forg, scale):
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

    del timesVaryVelx, timesVaryVely, timesVaryVelz
    return phix, phiy, phiz



def test(model, modelA, modelAA, test_loader):
    for batch_idx, AllInputs in enumerate(test_loader):
        [template, target, template_down, target_down, templateOrg, targetOrg] = AllInputs
        template, target = template.cuda(device=0), target.cuda(device=0)
        template_down, target_down = Variable(template_down.cuda(device=0)), Variable(target_down.cuda(device=0))
        targetOrg = Variable(targetOrg[0].cuda(device=0))
        templateOrg = Variable(templateOrg[0].cuda(device=0))
        with torch.no_grad():
            tv1 = model(template, target)
            phix1, phiy1, phiz1 = getFieldEnd(tv1, targetOrg, 0.25)
            forwardImage_s1 = getForwardImage(tv1, template[0,0,:,:,:], targetOrg, templateOrg, 0.25)

            #combine = torch.cat((forwardImage_s1.unsqueeze(0).unsqueeze(0), target[:,0,:,:,:].unsqueeze(1)), dim=1)

            tv2 = modelA(forwardImage_s1.unsqueeze(0).unsqueeze(0), target)
            phix2, phiy2, phiz2 = getFieldEnd(tv2, targetOrg, 0.5)
            forwardImage_s2 = getForwardImage(tv2, forwardImage_s1, targetOrg, templateOrg, 0.5)

            #combine = torch.cat((forwardImage_s2.unsqueeze(0).unsqueeze(0), images[:,1,:,:,:].unsqueeze(1)), dim=1)

            tv3 = modelAA(forwardImage_s2.unsqueeze(0).unsqueeze(0), target)
            phix3, phiy3, phiz3 = getFieldEnd(tv3, targetOrg, 1.0)
            
        return phix1, phiy1, phiz1, phix2, phiy2, phiz2, phix3, phiy3, phiz3
 

def test_2(model, modelA, modelAA, test_loader):
    for batch_idx, AllInputs in enumerate(test_loader):
        [template, target, template_down, target_down, templateOrg, targetOrg] = AllInputs
        template, target = template.cuda(device=0), target.cuda(device=0)
        template_down, target_down = Variable(template_down.cuda(device=0)), Variable(target_down.cuda(device=0))
        targetOrg = Variable(targetOrg[0].cuda(device=0))
        templateOrg = Variable(templateOrg[0].cuda(device=0))
        with torch.no_grad():
            tv1_xy = model(template, target)
            tv1_yx = model(target, template)
            tv1 = (tv1_xy-tv1_yx)/2
            #tv1 = tv1_xy
            phix1, phiy1, phiz1 = getFieldEnd(tv1, targetOrg, 0.25)
            forwardImage_s1 = getForwardImage(tv1, template[0,0,:,:,:], targetOrg, templateOrg, 0.25)

            #combine = torch.cat((forwardImage_s1.unsqueeze(0).unsqueeze(0), target[:,0,:,:,:].unsqueeze(1)), dim=1)

            tv2_xy = modelA(forwardImage_s1.unsqueeze(0).unsqueeze(0), target)
            tv2_yx = modelA(target, forwardImage_s1.unsqueeze(0).unsqueeze(0))
            tv2 = (tv2_xy - tv2_yx)/2
            #tv2 = tv2_xy
            phix2, phiy2, phiz2 = getFieldEnd(tv2, targetOrg, 0.5)
            forwardImage_s2 = getForwardImage(tv2, forwardImage_s1, targetOrg, templateOrg, 0.5)

            #combine = torch.cat((forwardImage_s2.unsqueeze(0).unsqueeze(0), images[:,1,:,:,:].unsqueeze(1)), dim=1)

            tv3_xy = modelAA(forwardImage_s2.unsqueeze(0).unsqueeze(0), target)
            #tv3_yx = modelAA(target, forwardImage_s2.unsqueeze(0).unsqueeze(0))
            #tv3 = (tv3_xy - tv3_yx)/2
            tv3 = tv3_xy
            phix3, phiy3, phiz3 = getFieldEnd(tv3, targetOrg, 1.0)
            
        return phix1, phiy1, phiz1, phix2, phiy2, phiz2, phix3, phiy3, phiz3
