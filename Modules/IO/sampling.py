from .loadData import load_imagesSinglePatient
from .loadData import sitkLoadImage
from .loadData import getRandIndexes
import numpy as np
import math
import random
import torch
import torch.utils.data as Data
import SimpleITK as sitk
import itertools

def imgnorm(img):
    i_max = np.max(img)
    i_min = np.min(img)
    norm = (img - i_min)/(i_max - i_min)
    return norm

class Dataset_epoch(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names, norm=False):
        'Initialization'
        super(Dataset_epoch, self).__init__()
        self.names = names
        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img_A = sitk.GetArrayFromImage(sitk.ReadImage(self.index_pair[step][0]))
        img_B = sitk.GetArrayFromImage(sitk.ReadImage(self.index_pair[step][1]))
        
        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float().unsqueeze(0), torch.from_numpy(imgnorm(img_B)).float().unsqueeze(0)
        else:
            return torch.from_numpy(img_A).float().unsqueeze(0), torch.from_numpy(img_B).float().unsqueeze(0)


class Predict_dataset(Data.Dataset):
    def __init__(self, fixed_list, move_list, fixed_label_list, move_label_list, norm=False):
        super(Predict_dataset, self).__init__()
        self.fixed_list = fixed_list
        self.move_list = move_list
        self.fixed_label_list = fixed_label_list
        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.move_list)

    def __getitem__(self, index):
        fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(self.fixed_list[index]))
        moved_img = sitk.GetArrayFromImage(sitk.ReadImage(self.move_list[index]))
        fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(self.fixed_label_list[index]))
        moved_label = sitk.GetArrayFromImage(sitk.ReadImage(self.move_label_list[index]))

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)
        fixed_label = torch.from_numpy(fixed_label)
        moved_label = torch.from_numpy(moved_label)

        output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                  'fixed_label': fixed_label.float(), 'move_label': moved_label.float(), 'index': index}
        return output


def getSamplesSubepoch(templateNames_Train,
                       targetNames_Train
                       ):

    print(" ... Get images for subEpoch ...")
    numTemplates_Epoch = len(templateNames_Train)
    numTargets_Epoch = len(targetNames_Train)
    randTemIdx = getRandIndexes(numTemplates_Epoch, numTemplates_Epoch)
    randTarIdx = getRandIndexes(numTargets_Epoch, numTargets_Epoch)
    temIdx = [] 
    tarIdx = []
    #for ii in range(len(randTemIdx)):
    for ii in range(20):
        temIdx.append(randTemIdx[ii])
        tarIdx.append(randTarIdx[ii])
    
    #print(temIdx)
    #print(tarIdx)
    templateSubjectsAll = []
    targetSubjectsAll = []
    templateOrgAll = []
    targetOrgAll = []
    
    templateSubjectsAll = []
    targetSubjectsAll = []
    templateOrgAll = []
    targetOrgAll = []

    for i_tem in range(0, len(temIdx)):
        templateSubject, templateOrg = sitkLoadImage(templateNames_Train[temIdx[i_tem]])
        templateSubjectsAll = templateSubjectsAll + [templateSubject]
        templateOrgAll = templateOrgAll + [templateOrg]

    for i_tar in range(0, len(tarIdx)):
        targetSubject,targetOrg= sitkLoadImage(targetNames_Train[tarIdx[i_tar]])
        targetSubjectsAll = targetSubjectsAll + [targetSubject]
        targetOrgAll = targetOrgAll + [targetOrg]

    return templateSubjectsAll, targetSubjectsAll, templateOrgAll, targetOrgAll

            

def getTestSubject(templateNames_Test,
                   targetNames_Test
                   ):
    #print(" ... Get images for Testing ...")
    templateSubjectsAll = []
    targetSubjectsAll = []
    templateOrgAll = []
    targetOrgAll = []
    
    
    [templateSubject,
         templateOrg] = sitkLoadImage(templateNames_Test)

    [targetSubject,
         targetOrg]= sitkLoadImage(targetNames_Test)
    
    return [imgnorm(templateSubject)], [imgnorm(targetSubject)], [templateOrg], [targetOrg]
