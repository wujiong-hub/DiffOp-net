import sys
import pdb
import numpy
from startTraining import startTraining
import torch.optim as optim
from Modules.shallowNetwork_att import MDRNet_r1, MDRNet_r2, MDRNet_r3
from Modules.Parsers.parsersUtils import parserConfigIni
import torch

""" To print function usage """
def printUsage(error_type):
    if error_type == 1:
        print(" ** ERROR!!: Few parameters used.")
    else:
        print(" ** ERROR!!: Asked to start with an already created network but its name is not specified.")
        
    print(" ******** USAGE ******** ")
    print(" --- argv 1: Name of the configIni file.")
    print(" --- argv 2: Number of start channels")
    print(" --- argv 3: Type of training:")
    print(" ------------- 0: Create a new model and start training")
    print(" ------------- 1: Use an existing model to keep on training (Requires an additional input with model name)")
    print(" --- argv 4: (Optional, but required if arg 2 is equal to 1) Network model name")
    


def networkTraining(argv):
    # Number of input arguments
    #    1: ConfigIniName
    #    2: TrainingType
    #             0: Create a new model and start training
    #             1: Use an existing model to keep on training (Requires an additional input with model name)
    #    3: (Optional, but required if arg 2 is equal to 1) Network model name
   
    # Do some sanity checks
    
    if len(argv) < 3:
        printUsage(1)
        sys.exit()
    
    configIniName = argv[0]
    nc = int(argv[1])
    trainingType  = argv[2]
    
    if trainingType == '1' and len(argv) == 3:
        printUsage(2)
        sys.exit()
        
    if len(argv)>3:
        networkModelName = argv[3]
    
    #print(nc)
    
    myParserConfigIni = parserConfigIni()
    myParserConfigIni.readConfigIniFile(configIniName,1) 
    # Creating a new model 
    if trainingType == '0':
        print (" ******************************************  CREATING NETWORK ******************************************")
        model_r1 = MDRNet_r1(start_channel=nc).cuda()
        model_r2 = MDRNet_r2(start_channel=nc).cuda()
        model_r3 = MDRNet_r3(start_channel=nc).cuda()

        model_r1.train()
        model_r2.train()
        model_r3.train()

        optimizer = torch.optim.Adam([
                {'params': model_r1.parameters()},
                {'params': model_r2.parameters()},
                {'params': model_r3.parameters()},
        ], lr=0.0001)
 

        startTraining(model_r1, model_r2, model_r3, nc, optimizer, configIniName, 0)
        print (" ******************************************  NETWORK CREATED ******************************************")
    else:
        print (" ******************************************  STARTING NETWORK TRAINING ******************************************")
        model_r1 = MDRNet_r1(start_channel=nc).cuda()
        model_r2 = MDRNet_r2(start_channel=nc).cuda()
        model_r3 = MDRNet_r3(start_channel=nc).cuda()

        optimizer = torch.optim.Adam([
                {'params': model_r1.parameters()},
                {'params': model_r2.parameters()},
                {'params': model_r3.parameters()},
        ], lr=0.0001)


        checkpoint = torch.load(networkModelName)
        #net.load_state_dict(checkpoint['model_state_dict'])
        model_r1.load_state_dict(checkpoint['model_r1_state_dict'])
        model_r2.load_state_dict(checkpoint['model_r2_state_dict'])
        model_r3.load_state_dict(checkpoint['model_r3_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        iterN = networkModelName.split('/')[-1]
        iterN = int(iterN.split('_')[-1].split('.')[0])

        model_r1.train()
        model_r2.train()
        model_r3.train()

        startTraining(model_r1, model_r2, model_r3, nc, optimizer, configIniName, iterN)
        print (" ******************************************  DONE  ******************************************")

if __name__ == '__main__':
   networkTraining(sys.argv[1:])
