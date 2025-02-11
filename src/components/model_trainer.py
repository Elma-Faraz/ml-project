from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
#from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models
import os
import sys

@dataclass
class ModelTrainerConfig:
    #defining path to save model.pkl file
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        #creating object for ModelTrainerConfig class to read model.pkl file path
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array,test_array):
        try:
            logging.info("split train and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            #all models used for training are defined below
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            
            # Define parameter grids for each model for hypertuning
            params = {
                "Linear Regression": {},
                "Lasso": {'alpha': [0.1, 1.0, 10.0]},
                "Ridge": {'alpha': [0.1, 1.0, 10.0]},
                "K-Neighbors Regressor": {'n_neighbors': [3, 5, 7, 9]},
                "Decision Tree": {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
                "Random Forest Regressor": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                "XGBRegressor": {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 1.0]
                }
            }
            
            #dictionary of each model with is score
            model_report:dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )
            
            #finding best model score
            best_model_score = max(sorted(model_report.values()))
            
            #finding best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            #if accuracy of best model is less than 60% than raising exception
            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            #saving best model to model.pkl file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)