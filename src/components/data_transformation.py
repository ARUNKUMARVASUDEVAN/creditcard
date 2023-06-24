import sys
from dataclasses import dataclass
from src.utils  import save_object
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.logger1 import logging
from src.exception import CustomException
import os


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Tranformation is initiated")
            numerical_columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            logging.info('Pipeline Initiated')
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns)
            ])

            return preprocessor
            logging.info('Pipeline completer')
           
        except Exception as e:
            logging.info("Error in Data Transformtion")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data is completed ")
            logging.info(f"Train DataFrame Head :\n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head :\n{test_df.head().to_string()}")

            logging.info("Obtaining Preprocessing object ")
            preprocessor_obj=self.get_data_transformation_object()

            target_column='default payment next month'
            drop_columns=['ID',target_column]

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            print('train df:',train_df.columns)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            print('test_df',test_df.columns)
            target_feature_test_df = test_df[test_df.columns.intersection([target_column])]
            
            #Transformtion
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            logging.info('Applying preprocesssing object on training and testing datasets. ')
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info('Preprocessor Pickele file saved')
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            logging.info('Error happened on the data_transformation step')

