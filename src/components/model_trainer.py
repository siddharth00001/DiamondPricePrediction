import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model
from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variable from train and test Dataset')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'ridge':Ridge(),
                'elastic':ElasticNet(),
                'treeRegressor':DecisionTreeRegressor()
            }
            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n'+'='*11)
            logging.info(f'Model report :{model_report}')
            # To get the best model score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            best_model = models[best_model_name]
            print(f'Best model Found ,Model name :{best_model_name},R2 Score :{best_model_score}')
            print('\n==================================================================')
            logging.info(f"Best Model Found, Model name :{best_model_name},R2 Score:{best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.shutdown()
        except Exception as e:
            logging.info('Exception occured at Model Training')
            logging.shutdown()
            raise CustomException(e,sys)
