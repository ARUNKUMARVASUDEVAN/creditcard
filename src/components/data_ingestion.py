import os
import sys
from src.exception import CustomException
from src.logger1 import logging
import pandas  as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method Starts')
        try:
            df=pd.read_excel('D:/Inuron/credit_analysis/notebooks/data/default of credit card clients.xls',header=1)
            df=df.drop_duplicates()
            
            
            logging.info("Dataset read as pandas DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train test split')

            train_set,test_set=train_test_split(df,test_size=0.30)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )




        
        except Exception as e:
            logging.info('Exception occured at Daata Ingestion Stage')
            raise CustomException(e,sys)
