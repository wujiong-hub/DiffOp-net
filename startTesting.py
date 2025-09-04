
import numpy as np
import time
import os
import pdb

from Modules.General.Evaluation import computeDice
from Modules.General.Utils import getImagesSet, dirMake
from Modules.IO.ImgOperations.imgOp import applyUnpadding
from Modules.IO.loadData import load_imagesSinglePatient
from Modules.IO.saveData import saveImageAsNifti
from Modules.IO.saveData import saveImageAsMatlab
from Modules.IO.sampling import *
from Modules.Parsers.parsersUtils import parserConfigIni

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import SimpleITK as sitk
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from Modules.dataload import MyDataset
from Modules.test import test, test_2
from Modules.imageApplyField import *
import lddmm 
import SimpleITK as sitk
import pystrum.pynd.ndutils as nd
import nibabel as nib


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

eval_det = AverageMeter()

def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field. 
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    disp = sitk.GetArrayFromImage(disp)
    # check inputs
    disp = disp.transpose(2, 1, 0, 3)
    #print(disp.shape)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

def dice(im1, atlas):
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    
    if np.max(unique_class)>2000:
        unique_class = [1002., 1003., 1005., 1006., 1007., 1008., 1009., 1011., 1012.,
            1013., 1014., 1015., 1016., 1017., 1018., 1021., 1022., 1024.,
            1025., 1028., 1029., 1030., 1031., 1034., 1035., 2002., 2003.,
            2005., 2006., 2007., 2008., 2009., 2011., 2012., 2013., 2014.,
            2015., 2016., 2017., 2018., 2021., 2022., 2024., 2025., 2028.,
            2029., 2030., 2031., 2034., 2035.]
    

    for i in unique_class:
        if (i == 0) or ((im1==i).sum()==0) or ((atlas==i).sum()==0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
    return dice/num_count


def imageRegistration(net1,
                      net2,
                      net3,
                      folderName,
                      i_d,
                      templateNames_Test,
                      targetNames_Test,
                      tem_names_Test,
                      tar_names_Test,
                      temLabelNames_Test,
                      tarLabelNames_Test,
                      batch_size,
                      task
                      ):

    #load the template image and target image

    [template_samplesAll,
     target_samplesAll,
     template_orgAll,
     target_orgAll] = getTestSubject(templateNames_Test[i_d],
                                     targetNames_Test[i_d]
                                     )

    template_samplesAll = np.array(template_samplesAll)
    target_samplesAll = np.array(target_samplesAll)
    template_orgAll = np.array(template_orgAll)
    target_orgAll = np.array(target_orgAll)
    
    
    test_data = MyDataset(template_samplesAll, target_samplesAll, template_orgAll, target_orgAll, 1.0)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=1)

    [phix1, phiy1, phiz1, phix2, phiy2, phiz2, phix3, phiy3, phiz3] = test_2(net1, net2, net3, test_loader)
    
    phix1 = phix1.cpu().data.numpy() 
    phiy1 = phiy1.cpu().data.numpy()
    phiz1 = phiz1.cpu().data.numpy()
    phix2 = phix2.cpu().data.numpy()
    phiy2 = phiy2.cpu().data.numpy()
    phiz2 = phiz2.cpu().data.numpy()
    phix3 = phix3.cpu().data.numpy()
    phiy3 = phiy3.cpu().data.numpy()
    phiz3 = phiz3.cpu().data.numpy()


    field1 = getField(phix1, phiy1, phiz1, targetNames_Test[i_d], 0.25)
    field2 = getField(phix2, phiy2, phiz2, targetNames_Test[i_d], 0.5)
    field3 = getField(phix3, phiy3, phiz3, targetNames_Test[i_d], 1.0)

    field_s2 = fieldApplyField(field2, field1, targetNames_Test[i_d])
    field_s3 = fieldApplyField(field3, field_s2, targetNames_Test[i_d])


    jac_det = jacobian_determinant_vxm(field_s3)
    eval_det.update(np.sum(jac_det <= 0) / np.prod((144,160,192)), 1)
    jet = np.sum(jac_det <= 0) / np.prod((144,160,192))
    #print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod((144,160,192))))

    
    alignedLabel_final = getTransformedImage(temLabelNames_Test[i_d], field_s3, targetNames_Test[i_d], useNearest=True)
    alignedLabel_s2 = getTransformedImage(temLabelNames_Test[i_d], field_s2, targetNames_Test[i_d], useNearest=True)
    targetLab = sitk.GetArrayFromImage(sitk.ReadImage(tarLabelNames_Test[i_d]))
    dice_final = dice(sitk.GetArrayFromImage(alignedLabel_final), targetLab)
    dice_org = dice(sitk.GetArrayFromImage(alignedLabel_s2), targetLab)
                
    #print(i_d, '\t', dice_final, '\t', dice_org, '\t',  np.sum(jac_det <= 0) / np.prod((144,160,192)))
    print(i_d, '\t', dice_final, '\t', dice_org, '\t')
    #return dice_final, dice_org
    
    
    
    # Generate folders to store the model
    BASE_DIR = os.getcwd()
    path_Temp = os.path.join(BASE_DIR, 'outputFiles')

    # For the predictions
    predlFolderName = os.path.join(path_Temp, folderName)
    predlFolderName = os.path.join(predlFolderName, 'Pred')
    if task == 0:
        predTestFolderName = os.path.join(predlFolderName, 'Validation')
    else:
        predTestFolderName = os.path.join(predlFolderName, 'Testing')

    '''
    dirMake(predTestFolderName)
    nameToSave = predTestFolderName + '/' + tem_names_Test[i_d].split('/')[-1].split('_img')[0] + '_' + tar_names_Test[i_d].split('/')[-1].split('_img')[0]

    fieldSaveDir = nameToSave + 'Warp.nii.gz' 
    sitk.WriteImage(field_s3, fieldSaveDir)
    #"******** field has been saved **********"
    
    
    alignedImage = getTransformedImage(templateNames_Test[i_d], field_s3, targetNames_Test[i_d], useNearest=False)
    templateImage = sitk.ReadImage(targetNames_Test[i_d])
    alignedImage.SetDirection(templateImage.GetDirection())
    alignedImage.SetOrigin(templateImage.GetOrigin())
    alignedImage.SetSpacing(templateImage.GetSpacing())
    alignedImageSaveDir = nameToSave + '.nii.gz'
    sitk.WriteImage(alignedImage, alignedImageSaveDir)

    #****aligned template has been saved *****
    
    templateImage = sitk.ReadImage(targetNames_Test[i_d])
    alignedLabel = getTransformedImage(temLabelNames_Test[i_d], field_s3, targetNames_Test[i_d], useNearest=True)
    alignedLabel.SetDirection(templateImage.GetDirection())
    alignedLabel.SetOrigin(templateImage.GetOrigin())
    alignedLabel.SetSpacing(templateImage.GetSpacing())
    alignedLabelSaveDir = nameToSave + '_seg.nii.gz'
    sitk.WriteImage(alignedLabel, alignedLabelSaveDir)
    #****aligned label has been saved *****
    
    gridimgpath ='../../UPenn/TraditionalRegMethods/script/grid_144_160_192.nii.gz'
    alignedgrid = getTransformedImage(gridimgpath, field_s3, targetNames_Test[i_d], useNearest=False)
    alignedgrid.SetDirection(templateImage.GetDirection())
    alignedgrid.SetOrigin(templateImage.GetOrigin())
    alignedgrid.SetSpacing(templateImage.GetSpacing())
    alignedgridSaveDir = nameToSave + 'grid.nii.gz'
    sitk.WriteImage(alignedgrid, alignedgridSaveDir)
                               
    nib.save(nib.Nifti1Image(nib.load(fieldSaveDir).get_fdata(), np.eye(4)), fieldSaveDir)
    nib.save(nib.Nifti1Image(nib.load(alignedImageSaveDir).get_fdata(), np.eye(4)), alignedImageSaveDir)
    nib.save(nib.Nifti1Image(nib.load(alignedLabelSaveDir).get_fdata(), np.eye(4)), alignedLabelSaveDir)
    nib.save(nib.Nifti1Image(nib.load(alignedgridSaveDir).get_fdata(), np.eye(4)), alignedgridSaveDir)

    
    alignedLabel1 = getTransformedImage(temLabelNames_Test[i_d], field_s2, targetNames_Test[i_d], useNearest=True)
    alignedLabel1.SetDirection(templateImage.GetDirection())
    alignedLabel1.SetOrigin(templateImage.GetOrigin())
    alignedLabel1.SetSpacing(templateImage.GetSpacing())
    alignedLabelSaveDir1 = nameToSave + '_2-label.nii.gz'
    sitk.WriteImage(alignedLabel1, alignedLabelSaveDir1)
    '''

    #everyROI    = computeDice(alignedLabelSaveDir, tarLabelNames_Test[i_d])
    #everyROIorg = computeDice(alignedLabelSaveDir1, tarLabelNames_Test[i_d])
    #print("meanDice:",i_d, np.mean(everyROI), np.mean(everyROIorg))
    
    
    return np.mean(dice_final), np.mean(dice_org), jet
    


""" Main registration function """
def startTesting(net1, net2, net3, configIniName) :

    print (" ******************************************  STARTING Registration ******************************************")

    print (" **********************  Starting segmentation **********************")
    myParserConfigIni = parserConfigIni()
    myParserConfigIni.readConfigIniFile(configIniName,2)
    

    print (" -------- Images to registration -------------")

    print (" -------- Reading Images names for registration -------------")
    
    # -- Get list of images used for testing -- #

    (templateNames_Test, tem_names_Test)     = getImagesSet(myParserConfigIni.TemplateImagesFolder,myParserConfigIni.indexesForTemplate)  # Templates
    (targetNames_Test, tar_names_Test)       = getImagesSet(myParserConfigIni.TargetImagesFolder,myParserConfigIni.indexesForTarget) # Targets

    (temLabelNames_Test, temLab_names_Test)  = getImagesSet(myParserConfigIni.TemplateLabelsFolder, myParserConfigIni.indexesForTemplate)
    (tarLabelNames_Test, tarLab_names_Test)  = getImagesSet(myParserConfigIni.TargetLabelsFolder, myParserConfigIni.indexesForTarget)

    print (" ================== Images for registration ================")

    #for i in range(0,len(tem_names_Test)):
    #    print(" template({}): {}  template Label: {} | target: {}  target Label: {}".\
    #          format(i,tem_names_Test[i], temLab_names_Test[i], tar_names_Test[i], tarLab_names_Test[i]))

    folderName            = myParserConfigIni.folderName
    batch_size            = myParserConfigIni.batch_size
    numberImagesToRegistration = len(templateNames_Test)
    allDsc, allOrgDsc, allJet = [], [], []
    scale =0.5
    
    for i_d in range(numberImagesToRegistration) :
        #print("**********************  Regist subject : {} to subject : {} ....total: {}/{}...**********************".format(tem_names_Test[i_d],tar_names_Test[i_d],str(i_d+1),str(numberImagesToRegistration)))
        dsc, orgDsc, jet = imageRegistration(net1,
                                        net2,
                                        net3,
                                        folderName, 
                                        i_d, 
                                        templateNames_Test,
                                        targetNames_Test,
                                        tem_names_Test,
                                        tar_names_Test,
                                        temLabelNames_Test,
                                        tarLabelNames_Test,
                                        batch_size,
                                        task = 1
                                        )
        allDsc.append(dsc)
        allOrgDsc.append(orgDsc)
        allJet.append(jet)

    print("{} {} {} {}".format(np.mean(allDsc), np.std(allDsc),np.round(np.mean(allJet)*100,3), np.round(np.std(allJet)*100,3)))
    
    print(" **************************************************************************************************** ")


