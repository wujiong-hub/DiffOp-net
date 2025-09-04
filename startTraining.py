import sys
import time
import numpy as np
import random
import math
import os
                  
from Modules.General.Utils import getImagesSet, dirMake
from Modules.IO.sampling import getSamplesSubepoch 
from Modules.Parsers.parsersUtils import parserConfigIni
from Modules.EnergyFunction import *
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pdb
from Modules.dataload import MyDataset, downsample_image
from Modules.IO.loadData import sitkLoadImage
from Modules.IO.sampling import Dataset_epoch, Predict_dataset 
import lddmm
import torch.utils.data as Data
from Modules.EnergyFunction import *
from startTesting import *
from natsort import natsorted
import csv, glob

energyLoss = EnergyFunction.apply 
torch.backends.cudnn.benchmark=True 


def save_checkpoint(state, save_dir, save_filename, max_model_num=10):
    torch.save(state, save_dir + save_filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0]) 
        model_lists = natsorted(glob.glob(save_dir + '*'))


def startTraining(net1, net2, net3, nc, optimizer, configIniName, inIter):
    print (" ************************************************  STARTING TRAINING **************************************************")
    print (" **********************  Starting training model (Reading parameters) **********************")

    myParserConfigIni = parserConfigIni()
    myParserConfigIni.readConfigIniFile(configIniName,1)

    print (" --- Do training in {} epochs with {} subEpochs each...".format(myParserConfigIni.numberOfEpochs, myParserConfigIni.numberOfSubEpochs))
    print ("-------- Reading Images names used in training/-------------")

    # -- Get list of images used for training -- #
    (templateNames_Train, tem_names_Train)     = getImagesSet(myParserConfigIni.temImagesFolder,myParserConfigIni.indexesForTemplate)  # Images
    (targetNames_Train, tar_names_Train)       = getImagesSet(myParserConfigIni.tarImagesFolder,myParserConfigIni.indexesForTarget) # Ground truth

    (templateNames_Valid, tem_names_Valid)     = getImagesSet(myParserConfigIni.temImagesFolder_val,myParserConfigIni.indexesForTemplate_val)  # Images
    (targetNames_Valid, tar_names_Valid)       = getImagesSet(myParserConfigIni.tarImagesFolder_val,myParserConfigIni.indexesForTarget_val) # Ground truth
    (templateLabels_Valid, tem_labels_Valid)     = getImagesSet(myParserConfigIni.temLabelsFolder_val,myParserConfigIni.indexesForTemplate_val)  # Images
    (targetLabels_Valid, tar_labels_Valid)       = getImagesSet(myParserConfigIni.tarLabelsFolder_val,myParserConfigIni.indexesForTarget_val) # Ground truth
    
    # Print names``
    print (" ================== Images for training ================")
    #for i in range(0,len(tem_names_Train)):
    #    print(" Template Image({}): {} | Target Image {}".format(i,tem_names_Train[i],tar_names_Train[i]))

    numberOfEpochs = myParserConfigIni.numberOfEpochs
    numberOfSubEpochs = myParserConfigIni.numberOfSubEpochs
    batch_size = myParserConfigIni.batch_size
    folderName = myParserConfigIni.folderName 

    r1_scale, r2_scale, r3_scale = 0.25, 0.5, 1.0
    r1_alpha, r2_alpha, r3_alpha = 0.005,0.001,0.001
    alpha_str = 511
    sigma = 1e-6
    iteration = 150000
    
    model_dir = './logs/LDDMM_alpha_{}_sigma_{}_ch_{}/'.format(alpha_str, sigma, nc)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    csv_name = './logs/LDDMM_alpha_{}_sigma_{}_ch_{}.csv'.format(alpha_str, sigma, nc)
    f = open(csv_name, 'w')
    with f:
        fnames = ['Index','Dice']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()


    exampleImageData, org = sitkLoadImage(templateNames_Train[0])
    exampleImageData = downsample_image(exampleImageData, r1_scale)
    exampleImage = torch.from_numpy(exampleImageData)
    exampleImage = exampleImage.cuda()
    r1_LKernel, r1_AKernel = lddmm.KernelInitialize(exampleImage, r1_alpha, 1)

    exampleImageData, org = sitkLoadImage(templateNames_Train[0])
    exampleImageData = downsample_image(exampleImageData, r2_scale)
    exampleImage = torch.from_numpy(exampleImageData)
    exampleImage = exampleImage.cuda()
    r2_LKernel, r2_AKernel = lddmm.KernelInitialize(exampleImage, r2_alpha, 1)

    exampleImageData, org = sitkLoadImage(templateNames_Train[0])
    exampleImage = torch.from_numpy(exampleImageData)
    exampleImage = exampleImage.cuda()
    r3_LKernel, r3_AKernel = lddmm.KernelInitialize(exampleImage, r3_alpha, 1)


    training_generator = Data.DataLoader(Dataset_epoch(templateNames_Train, norm=True), batch_size=1,
                                         shuffle=False, num_workers=1)
    
    validation_generator = Data.DataLoader(Predict_dataset(targetNames_Valid, templateNames_Valid, targetLabels_Valid, templateLabels_Valid, norm=True), batch_size=1,
                                         shuffle=False, num_workers=1)


    r1_Lkernel = Variable(r1_LKernel.cuda())
    r1_Akernel = Variable(r1_AKernel.cuda())
    r2_Lkernel = Variable(r2_LKernel.cuda())
    r2_Akernel = Variable(r2_AKernel.cuda())
    r3_Lkernel = Variable(r3_LKernel.cuda())
    r3_Akernel = Variable(r3_AKernel.cuda())
    r1_voxelVolum = torch.tensor([1/(r1_scale*r1_scale*r1_scale)])
    r2_voxelVolum = torch.tensor([1/(r2_scale*r2_scale*r2_scale)])
    r3_voxelVolum = torch.tensor([1/(r3_scale*r3_scale*r3_scale)])
    Sigma = torch.tensor([sigma])
    r1_scale = torch.tensor([r1_scale])
    r2_scale = torch.tensor([r2_scale])
    r3_scale = torch.tensor([r3_scale])
    targetOrg = Variable(torch.tensor([-80.0,112.0,96.0]).cuda())
    templateOrg = Variable(torch.tensor([-80.0,112.0,96.0]).cuda())

    step = inIter

    while step <= iteration:
        for X, Y in training_generator:
            src = X.cuda().float() 
            tgt = Y.cuda().float()


            tv1 = net1(src, tgt)
            r1_forwardImage = getForwardImage(tv1, src[0,0,:,:,:], targetOrg, templateOrg, r1_scale[0])  

            tv2 = net2(r1_forwardImage.unsqueeze(0).unsqueeze(0), tgt)
            r2_forwardImage = getForwardImage(tv2, r1_forwardImage, targetOrg, templateOrg, r2_scale[0])  

            tv3 = net3(r2_forwardImage.unsqueeze(0).unsqueeze(0),tgt)
            r3_forwardImage = getForwardImage(tv3, r2_forwardImage, targetOrg, templateOrg, r3_scale[0])  
            
            loss_1 = energyLoss(tv1, r1_forwardImage, tgt[0,0,:,:,:], r1_Lkernel, r1_Akernel, targetOrg, r1_voxelVolum[0], r1_scale[0], Sigma[0])
            loss_2 = energyLoss(tv2, r2_forwardImage, tgt[0,0,:,:,:], r2_Lkernel, r2_Akernel, targetOrg, r2_voxelVolum[0], r2_scale[0], Sigma[0])
            loss_3 = energyLoss(tv3, r3_forwardImage, tgt[0,0,:,:,:], r3_Lkernel, r3_Akernel, targetOrg, r3_voxelVolum[0], r3_scale[0], Sigma[0])
            
            optimizer.zero_grad() 
            loss = loss_1 + loss_2 + loss_3
            loss.backward()
            optimizer.step()


            del tv1, r1_forwardImage, tv2, r2_forwardImage, tv3, r3_forwardImage

            print("Energy {} | loss: {:.3e} loss_r1: {:.3e} loss_r2: {:.3e} loss_r3: {:.3e}".format(step, loss.item(), loss_1.item(), loss_2.item(), loss_3.item()))
            optimizer.step()
            
            if ((step) % 500 == 0): 
                BASE_DIR = os.getcwd()
                #path_Temp = os.path.join(BASE_DIR,'outputFiles')
                #netFolderName = os.path.join(path_Temp,'models_'+str(numberVels)+'v')
                #netFolderName  = os.path.join(netFolderName,'Networks')
                #dirMake(netFolderName)


                numberImagesToRegistration = len(templateNames_Valid)
                allDsc, allOrgDsc = [], []
                for i_d in range(numberImagesToRegistration) :
                    dsc, orgDsc, _ = imageRegistration(net1,
                                                    net2,
                                                    net3,
                                                    folderName,
                                                    i_d,
                                                    templateNames_Valid,
                                                    targetNames_Valid,
                                                    tem_names_Valid,
                                                    tar_names_Valid,
                                                    templateLabels_Valid,
                                                    targetLabels_Valid,
                                                    batch_size,
                                                    task=1
                                                    )

                
                    allDsc.append(dsc)
                    allOrgDsc.append(orgDsc)
            
                print("The final meanDsc is :{} orgDsc: {}".format(np.mean(allDsc), np.mean(allOrgDsc)))

                save_model = {
                    'model_r1_state_dict': net1.state_dict(),
                    'model_r2_state_dict': net2.state_dict(),
                    'model_r3_state_dict': net3.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }

                modelname = 'DiceVal_{:.4f}_Epoch_{:04d}.pth'.format(np.round(np.mean(allDsc),4), step)
                csv_dice = np.mean(allDsc)
                f = open(csv_name, 'a')
                with f:
                    writer = csv.writer(f)
                    writer.writerow([step, csv_dice])
                save_checkpoint(save_model, model_dir, modelname)

            step += 1

            if step > iteration:
                break

    print("................ The whole Training is done..............")
    print(" ************************************************************************************ ")
