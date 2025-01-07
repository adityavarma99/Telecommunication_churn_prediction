import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass                        # dataclass helps us to build class variable

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

## Creating input class to store the train, test and raw data in artifacts folder, it stores using class DataIngestionConfig
## @dataclass is to define the class variable like train_data_path, test and raw data path
@dataclass 
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")           # data ingestion will save the train.csv into train_data_path
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/telecommunications_churn.csv')                                     # read the data
            logging.info('Read the dataset as dataframe')

            ## creating folders in artifacts train, test and raw
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)       # converted raw data into csv file

            logging.info("Train test split initiated") 
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)        # train test split     

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) # saving the train file

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)   # saving the test file


            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,                                # returning train and test data path for data transformation
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)








