import os 
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd 
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer

@dataclass#decorator
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts',"train.csv")#all output files are stored in artifacts. 
    test_data_path:str = os.path.join('artifacts',"test.csv")
    raw_data_path:str = os.path.join('artifacts',"raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):#we can also create mongodb or mysql client to read from databases  in utils 
        logging.info("Entered the dataingestion method or component")
        try:
            df = pd.read_csv('notebook\stud.csv')#copy relative path
            logging.info("Read the dataset as dataframe")

            #we want to create a folder named artifacts naturally within the mlops project
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) #os.path.dirname - creates folder with name off train path.
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Train test split initiated')
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of data is completed.')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except:
            raise CustomException(e,sys)
if __name__ == '__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()#once executed artifact folder will be created. 

    data_transformation  = DataTransformation()
    #data_transformation.initiate_data_transformation(train_data,test_data) # uncomment when we want to run data transformation
    #for running model trainer - use below
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)
    #creating object for model trainer now below line
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

