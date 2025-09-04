import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as ni
from scipy import ndimage

def gaussian_blur(label, sigma=2, size=3): #3 5   2,3
    # compute the truncate using size
    t = (((size - 1)/2)-0.5)/sigma
    smoothLabel = ndimage.gaussian_filter(label,sigma, truncate=t)
    return smoothLabel

def downsample_image(I,down):
    ''' downsample an image by averaging
    down should either be a triple, or a single number
    '''
    down = int(1.0/down)
    try:
        # check if its an iterable with 3 elements
        d0 = down[2]
    except TypeError:
        down = [down,down,down]
    down = np.array(down)
    nx = I.shape
    nxd = nx//down
    Id = np.zeros(nxd,dtype=np.float32)
    for i in range(down[0]):
        for j in range(down[1]):
            for k in range(down[2]):
                Id += I[i:nxd[0]*down[0]:down[0],j:nxd[1]*down[1]:down[1],k:nxd[2]*down[2]:down[2]]
    Id = Id/down[0]/down[1]/down[2]
    return Id

def guassian_label(templateData, targetData):
    count = 0
    m = targetData.shape[0]
    n = targetData.shape[1]
    z = targetData.shape[2]
    AllLabels_tem = np.zeros((len(np.unique(templateData))-1, m, n, z), np.float32)
    AllLabels_tar = np.zeros((len(np.unique(templateData))-1, m, n, z), np.float32)
    for iLabel in np.unique(templateData)[1:]:
        AllLabels_tem[count,:,:,:] = gaussian_blur((templateData == iLabel).astype(np.float32))
        AllLabels_tar[count,:,:,:] = gaussian_blur((targetData == iLabel).astype(np.float32))
        #AllLabels_tem[count,:,:,:] = (templateData == iLabel).astype(np.float32)
        #AllLabels_tar[count,:,:,:] = (targetData == iLabel).astype(np.float32)
        count = count+1
    return [AllLabels_tem, AllLabels_tar]

class MyDataset(Dataset):       
    def __init__(self, templateAll, targetAll, temOrgAll, tarOrgAll, down):
        
        self.templates = templateAll
        self.targets = targetAll
        self.temOrgs = temOrgAll
        self.tarOrgs = tarOrgAll
        self.down = down

    def __getitem__(self, index):
        template = self.templates[index,:,:,:]
        target = self.targets[index,:,:,:]
        if self.down < 1:
            template_down = downsample_image(template, self.down)
            target_down = downsample_image(target, self.down)
            template_down = gaussian_blur(template_down)
            target_down = gaussian_blur(target_down)
        else:
            template_down = template
            target_down = target

        template = torch.Tensor(template)
        target   = torch.Tensor(target)
        template = torch.unsqueeze(template,0)
        target   = torch.unsqueeze(target, 0)
        
        template_down = torch.Tensor(template_down)
        target_down   = torch.Tensor(target_down)
        template_down = torch.unsqueeze(template_down,0)
        target_down   = torch.unsqueeze(target_down, 0)

        templateOrg = self.temOrgs[index]
        targetOrg = self.tarOrgs[index]
        templateOrg = torch.Tensor(templateOrg)
        targetOrg = torch.Tensor(targetOrg)

        #return [torch.cat((template,target),0), torch.cat((template_down, target_down),0), templateOrg, targetOrg]
        return [template, target, template_down, target_down, templateOrg, targetOrg]
    
    def __len__(self):

        return len(self.templates)

