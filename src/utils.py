import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path, obj):
    try:
       
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        # Log object information
        logging.info(f"Saving object to {file_path}")
        logging.info(f"Object to be saved: {obj}")
        
        # Save the object using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully to {file_path}")

    except Exception as e:
        logging.error(f"Error occurred while saving object to {file_path}: {str(e)}")
        raise CustomException(e, sys)
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores 
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report
    
    except Exception as e:
            logging.info('Exception occured during model training')
            raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)