import pandas as pd
import numpy as np
import pickle
import sys
import os
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            # Predicting the output
            y_pred = model.predict(X_test)
            score = r2_score(y_test,y_pred)
            report[list(models.keys())[i]] = score
        return report
    except Exception as e:
        logging.info('Exception occured during model training')
        logging.shutdown()
        raise CustomException(e,sys)