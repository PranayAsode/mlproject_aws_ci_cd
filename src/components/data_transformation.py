import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.utils import save_object



# Creating inputs required for data transformation

@dataclass
class DataTransformatinConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformatinConfig()

    
    def get_data_transformation_pipeline(self):

        """
        This function is responsible for creating data transformation pipeline
        """

        try:
            logging.info("Data Transformation Pipeline Initiated")

            numerical_features = ['reading_score', 'writing_score']

            categorical_features = ['gender', 
                                    'race_ethnicity', 
                                    'parental_level_of_education', 
                                    'lunch',
                                    'test_preparation_course']
            logging.info(f"Numerical columns: {numerical_features}")
            logging.info(f"Categorical columns: {categorical_features}")
            

            logging.info("Numerical pipeline initiated")
            numerical_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="median")),
                    ("StandardScaler",StandardScaler())
                ]
            )
            logging.info("Numerical pipeline completed")


            logging.info("Categorical pipeline initiated")
            categorical_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoder",OneHotEncoder()),
                ]
            )
            logging.info("Categorical pipeline completed")


            logging.info("Column transformation intiated")
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline",numerical_pipeline,numerical_features),
                    ("categorical_pipeline",categorical_pipeline,categorical_features)
                ]
            )
            logging.info("Column transformation completed")
            logging.info("Data Transformation Pipeline Completed")

            return preprocessor
        
        except Exception as e:
            logging.info("Exception as occured")
            logging.info(CustomException(e,sys))
            raise CustomException(e,sys)


    def initiate_data_tranformation(self,train_path,test_path):
        """
        This function is responsible for data transformation
        and give train and test array and preprocessor file path
        """

        logging.info("Data Tranformation Intiated")

        try:

            logging.info("Reading of train and test data initiated")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading of train and test data completed")

            target_column_name = "math_score"

            # Creating X_train and y_train 
            input_features_train_df = train_df.drop(columns=[target_column_name],axis = True)
            target_feature_train_df = train_df[target_column_name]

            # Creating X_test and y_test
            input_features_test_df = test_df.drop(columns=[target_column_name],axis = True)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Creating preprocessor object")
            preprocessor_obj = self.get_data_transformation_pipeline()

            logging.info("Applying preprocessing object on train and test dataframe")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_features_test_df)

            # Concatinating train and test input features arr and target feature arr

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info("Preprocessing completed")

            # Using save_object function from utils.py file 
            save_object(self.data_transformation_config.preprocessor_obj_file_path,preprocessor_obj)
            logging.info("Preprocessor object saved")
            

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            logging.info("Exception has occured")
            logging.info(CustomException(e,sys))
            raise CustomException(e,sys)
        










