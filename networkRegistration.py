import sys
from startTesting import startTesting
from Modules.shallowNetwork_att import MDRNet_r1, MDRNet_r2, MDRNet_r3
import torch
import numpy as np

def calc_param_size(model):
    '''
    Show the memory cost of model.parameters, in MB. 
    '''
    return np.sum(np.prod(v.size()) for v in model.parameters())*4e-6
    
def printUsage(error_type):
    if error_type == 1:
        print(" ** ERROR!!: Few parameters used.")
    else:
        print(" ** ERROR!!: Asked to start with an already created network but its name is not specified.")
        
    print(" ******** USAGE ******** ")
    print(" --- argv 1: Name of the configIni file.")
    print(" --- argv 2: Number of channels")
    print(" --- argv 3: Network model name")


def networkRegistration(argv): 
    # Number of input arguments
    #    1: ConfigIniName (for segmentation)
    #    2: Network model name
   
    # Some sanity checks
    
    if len(argv) < 3:
        printUsage(1)
        sys.exit()

    configIniName = argv[0]
    nc = int(argv[1])
    networkModelName = argv[2]


    model_r1 = MDRNet_r1(start_channel=nc).cuda()
    model_r2 = MDRNet_r2(start_channel=nc).cuda()
    model_r3 = MDRNet_r3(start_channel=nc).cuda()

    # --------------- Load my FCN object  --------------- 
    print (" ... Loading model from {}".format(networkModelName))
    checkpoint = torch.load(networkModelName, map_location="cuda:0")
    print (" ... Network architecture successfully loaded....")
    
    print('Param size = {:.3f} MB'.format(calc_param_size(model_r1)))
    print('Param size = {:.3f} MB'.format(calc_param_size(model_r2)))
    print('Param size = {:.3f} MB'.format(calc_param_size(model_r3)))

    from torchinfo import summary
    x = torch.randn(1, 1, 192, 160, 144).cuda()
    y = torch.randn(1, 1, 192, 160, 144).cuda()
    input = torch.cat((x,y),dim=1)
    #print(summary(model_r1, input_data=(x,y)))
    #print(summary(model_r2, input_data=(x,y)))
    #print(summary(model_r3, input_data=(x,y)))
    
    model_r1.load_state_dict(checkpoint['model_r1_state_dict'])
    model_r2.load_state_dict(checkpoint['model_r2_state_dict'])
    model_r3.load_state_dict(checkpoint['model_r3_state_dict'])
    model_r1.eval()
    model_r2.eval()
    model_r3.eval()
    startTesting(model_r1, model_r2, model_r3, configIniName)
    print(" ***************** REGISTRATION DONE!!! ***************** ")
  
   
   
if __name__ == '__main__':
   networkRegistration(sys.argv[1:])
