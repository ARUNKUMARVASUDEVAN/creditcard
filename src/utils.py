import os
import sys

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,f1_score,precision_score,recall_score

from src.exception import CustomException
from src.logger1 import logging

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            y_test_pred=model.predict(X_test)
            test_model_score=roc_auc_score(y_test,y_test_pred)
    
            confusion=confusion_matrix(y_test,y_test_pred)
            report[list(models.keys())[i]]=test_model_score,confusion
            return report 
        
    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as obj:
            return pickle.load(obj)
        
    except Exception as e:
        logging.info('Exception occured in load_object function utils')
        raise CustomException(e,sys)
        
    