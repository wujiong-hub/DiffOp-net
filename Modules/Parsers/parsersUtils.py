import configparser
import json
import os

# -------- Parse parameters to create the network model -------- #
class parserConfigIni(object):
   def __init__(_self):
      _self.networkName = []
      
   #@staticmethod
   def readConfigIniFile(_self,fileName,task):
      # Task: 0-> Generate model
      #       1-> Train model
      #       2-> Testing

      def createModel():
          print (" --- Creating model (Reading parameters...)")
          _self.readModelCreation_params(fileName)
      def trainModel():
          print (" --- Training model (Reading parameters...)")
          _self.readModelTraining_params(fileName)
      def testModel():
          print (" --- Testing model (Reading parameters...)")
          _self.readModelTesting_params(fileName)
       
        # TODO. Include more optimizers here
      optionsParser = {0 : createModel,
                       1 : trainModel,
                       2 : testModel}
      optionsParser[task]()

   # Read parameters to Generate model

      # TODO: Do some sanity checks

   # Read parameters to TRAIN model
   def readModelTraining_params(_self,fileName) :
      ConfigIni = configparser.ConfigParser()
      ConfigIni.read(fileName)

      # Get training image names
      # Paths
      _self.temImagesFolder             = ConfigIni.get('Training Images','temImagesFolder')
      _self.tarImagesFolder             = ConfigIni.get('Training Images', 'tarImagesFolder')
      _self.folderName                  = ConfigIni.get('Training Images', 'folderName')
      _self.indexesForTemplate          = json.loads(ConfigIni.get('Training Images','indexesForTemplate'))
      _self.indexesForTarget        	 = json.loads(ConfigIni.get('Training Images', 'indexesForTarget'))
      
      _self.batch_size                        = json.loads(ConfigIni.get('Training Parameters','batch_size'))
      _self.numberOfEpochs                    = json.loads(ConfigIni.get('Training Parameters','number of Epochs'))
      _self.numberOfSubEpochs                 = json.loads(ConfigIni.get('Training Parameters','number of SubEpochs'))

      _self.temImagesFolder_val             = ConfigIni.get('Validation Images', 'temImagesFolder')
      _self.tarImagesFolder_val             = ConfigIni.get('Validation Images', 'tarImagesFolder')
      _self.temLabelsFolder_val             = ConfigIni.get('Validation Images', 'temLabelsFolder')
      _self.tarLabelsFolder_val             = ConfigIni.get('Validation Images', 'tarLabelsFolder')
      _self.indexesForTemplate_val          = json.loads(ConfigIni.get('Validation Images','indexesForTemplate'))
      _self.indexesForTarget_val        	  = json.loads(ConfigIni.get('Validation Images', 'indexesForTarget'))

   def readModelTesting_params(_self,fileName) :
      ConfigIni = configparser.ConfigParser()
      ConfigIni.read(fileName)
 
      _self.TemplateImagesFolder             = ConfigIni.get('Testing Images','TemplateImagesFolder')
      _self.TargetImagesFolder               = ConfigIni.get('Testing Images', 'TargetImagesFolder')
      
      _self.folderName               = ConfigIni.get('Testing Images','folderName')
      _self.indexesForTemplate       = json.loads(ConfigIni.get('Testing Images','indexesForTemplate'))
      _self.indexesForTarget         = json.loads(ConfigIni.get('Testing Images', 'indexesForTarget'))
     
      _self.batch_size               = json.loads(ConfigIni.get('Testing Images','batch_size'))

      _self.TemplateLabelsFolder     = ConfigIni.get('Testing Images', 'TemplateLabelsFolder')
      _self.TargetLabelsFolder       = ConfigIni.get('Testing Images', 'TargetLabelsFolder')
      #_self.indexesForTemLab         = json.loads(ConfigIni.get('Testing Images', 'indexesForTemLab'))
      #_self.indexesForTarLab         = json.loads(ConfigIni.get('Testing Images', 'indexesForTarLab'))


