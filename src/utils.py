import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

#code to save files
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}
        
        #iterating over all the models in models dictionary
        for i in range(len(list(models))):
            #getting a model
            model = list(models.values())[i]
            
            #hyperparameter code started
            #getting hyperparameter for the model
            param = params[list(models.keys())[i]]
            
            #performing hyperparamet tuning using GridSearchCV
            grid_search =GridSearchCV(model,param,cv=3)
            grid_search.fit(X_train,y_train)
            
            #updating model with best hyperparameter found during hyperparameter tuning
            model.set_params(**grid_search.best_params_)
            #hyperparameter tuning code ended
            
            #model training
            model.fit(X_train,y_train)
            
            #predicting the output on training data
            y_train_pred = model.predict(X_train)
            
            #predicting the output on test data
            y_test_pred = model.predict(X_test)
            
            #finding the accuracy usinf r2 square
            train_model_score = r2_score(y_train,y_train_pred)
            
            test_model_score = r2_score(y_test,y_test_pred)
            
            #returning the dictionary with model and score
            report[list(models.keys())[i]] = test_model_score
            
            return report
    except Exception as e:
        raise CustomException(e,sys)

