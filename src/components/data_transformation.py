# Importing relevant Libraries
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys,os
#Train Test Split Data
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object
## Data transformation
@dataclass
class DataTransformationconfig:
    preprocesser_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

## Data Transformation class

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()
    def get_data_transformation_object(self):
        try:
            logging.info('Data transformation intitiated')
            ## Getting numerical and categorical Features
            numerical_cols = ['carat','depth','table','x','y','z']
            categorical_cols = ['cut','color','clarity']
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair','Good','Very Good','Premium','Ideal']
            color_categories = ['D','E','F','G','H','I','J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            logging.info('Pipeline Started')
            ## Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            ## categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )
            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            return preprocessor
            logging.info('Pipeline Completed')
            logging.shutdown()
        except Exception as e:
            logging.info('Error in Data Transformation')
            logging.shutdown()
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            #Reading train and test data
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info('Read train and test data completed')
            logging.info(f'Train dataframe head :\n {train_data.head().to_string()}')
            logging.info(f'Test dataframe head: \n {test_data.head().to_string()}')
            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()

            # Putting features into dependent and independent features
            target='price'
            drop_col = [target,'id']
            input_feature_train_df  = train_data.drop(drop_col,axis='columns')
            target_feature_train_df = train_data[target]
            input_feature_test_df  = test_data.drop(drop_col,axis='columns')
            target_feature_test_df = test_data[target]
            #Preprocessing Data
            input_data_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_data_test_array = preprocessing_obj.transform(input_feature_test_df)
            logging.info('Applying preprocessing object on training and testing datasets')
            train_arr = np.c_[input_data_train_array,np.array(target_feature_train_df)]
            test_arr = np.c_[input_data_test_array,np.array(target_feature_test_df)]
            print(self.data_transformation_config.preprocesser_obj_file_path)
            save_object(
                file_path=self.data_transformation_config.preprocesser_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info('Processor Pickle is created and saved')
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesser_obj_file_path
            )
            logging.shutdown()

        except Exception as e:
            raise CustomException(e,sys)

