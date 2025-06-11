import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerCofig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerCofig()

    def initiate_model_trainer(self,train_array,test_array):
        """
        This function is responsible for model training and evaluation.
        This function gives model name and model score
        
        """
        try:
            logging.info("Creating train and test ")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Creating train and test data completed")
    
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            logging.info("Model Evaluation initiated")
            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            # Getting best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # Getting best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model")
            logging.info("Model evaluation completed")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            model_pred_values = best_model.predict(X_test)

            R2_score = r2_score(y_test,model_pred_values)

            logging.info(f"Best Model is {best_model_name} and r2 score is {R2_score}")

            return (best_model_name, R2_score)


        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e,sys)




