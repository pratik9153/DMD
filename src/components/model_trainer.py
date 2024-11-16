import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor  
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            # Splitting dependent and independent variables
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features for training
                train_array[:, -1],   # Target for training
                test_array[:, :-1],   # Features for testing
                test_array[:, -1]     # Target for testing
            )

            # Define models
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'Elasticnet': ElasticNet(),
                'DecisionTree': DecisionTreeRegressor(),
                'RandomForest': RandomForestRegressor() 
            }

            # Evaluate models
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
            logging.info(f'Model Report: {model_report}')

            # Get best model
            best_model_score = max(model_report.values())  # Highest R² score
            best_model_name = [key for key, value in model_report.items() if value == best_model_score][0]
            best_model = models[best_model_name]

            logging.info(f'Best Model Found: {best_model_name}, R² Score: {best_model_score}')
            print(f'Best Model Found: {best_model_name}, R² Score: {best_model_score}')

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f'Model saved successfully to {self.model_trainer_config.trained_model_file_path}')

        except Exception as e:
            logging.error(f'Exception occurred during model training: {str(e)}')
            raise CustomException(e, sys)
