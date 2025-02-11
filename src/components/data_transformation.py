# from src.components.data_ingestion import DataIngestion
# from src.exception import CustomException
# from src.logger import logging
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# import pandas as pd
# import numpy as np
# import sys
# import os
# # Modelling
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
# from sklearn.svm import SVR
# from sklearn.linear_model import LinearRegression, Ridge,Lasso
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# from sklearn.model_selection import RandomizedSearchCV
# #from catboost import CatBoostRegressor
# from xgboost import XGBRegressor
# import warnings

# def training_data_preprocessing(train_data_path):
#     logging.info("reading train data")
#     df = pd.read_csv(train_data_path)
#     logging.info(df.head(5))
    
#     logging.info("taking features from data")
#     X_train = df.drop(columns=['math_score'], axis=1)
#     logging.info(f"features of training data {X_train.head(5)}")
    
#     y_train = df['math_score']
#     logging.info(f"target of training data {y_train.head(5)}")
    
#     logging.info("splitting features into numerical and categorical features and preprocessing them")
#     num_features = X_train.select_dtypes(exclude='object').columns
#     cat_features = X_train.select_dtypes(include='object').columns
    
#     numeric_transformer = StandardScaler()
#     cat_transformer = OneHotEncoder()
    
#     preprocessor = ColumnTransformer(
#         [
#             ("OneHotEncoder", cat_transformer, cat_features),
#             ("StandardScaler", numeric_transformer, num_features)
#         ]
#     )
    
#     logging.info("fetures preprocessed successfully")
    
#     X_train = preprocessor.fit_transform(X_train)
#     logging.info(f"shape of features after preprocessing {X_train.shape}")
    
#     return X_train,y_train

# def preprocessing_test_data(test_data_path):
#     logging.info("reading test data")
#     df = pd.read_csv(test_data_path)
#     logging.info(df.head(5))
    
#     logging.info("taking features from data")
#     X_test = df.drop(columns=['math_score'], axis=1)
#     logging.info(f"features of training data {X_test.head(5)}")
    
#     y_test = df['math_score']
#     logging.info(f"target of training data {y_test.head(5)}")
    
#     logging.info("splitting features into numerical and categorical features and preprocessing them")
#     num_features = X_test.select_dtypes(exclude='object').columns
#     cat_features = X_test.select_dtypes(include='object').columns
    
#     numeric_transformer = StandardScaler()
#     cat_transformer = OneHotEncoder()
    
#     preprocessor = ColumnTransformer(
#         [
#             ("OneHotEncoder", cat_transformer, cat_features),
#             ("StandardScaler", numeric_transformer, num_features)
#         ]
#     )
    
#     logging.info("fetures preprocessed successfully")
    
#     X_test = preprocessor.transform(X_test)
#     logging.info(f"shape of features after preprocessing {X_test.shape}")
    
#     return X_test,y_test
    

# def evaluate_model(true, predicted):
#     mae = mean_absolute_error(true, predicted)
#     mse = mean_squared_error(true, predicted)
#     rmse = np.sqrt(mean_squared_error(true, predicted))
#     r2_square = r2_score(true, predicted)
#     return mae, rmse, r2_square

# def data_training(train_data_path, test_data_path, models):
#     logging.info("training started")
#     try:
#         X_train, y_train = training_data_preprocessing(train_data_path)
#         X_test, y_test = preprocessing_test_data(test_data_path)
        

#         for i in range(len(list(models))):
#             model = list(models.values())[i]
#             model.fit(X_train, y_train) # Train model

#             # Make predictions
#             y_train_pred = model.predict(X_train)
#             y_test_pred = model.predict(X_test)
            
#             # Evaluate Train and Test dataset
#             result_train = evaluate_model(y_train, y_train_pred)

#             result_test = evaluate_model(y_test, y_test_pred)
            
#             return result_train, result_test
        
#     except Exception as e:
#         raise CustomException(e,sys)
    
# if __name__ == "__main__":
#     dataingestion_obj = DataIngestion()

#     train_data_path, test_data_path = dataingestion_obj.initiate_data_ingestion()
#     print("train data path",train_data_path)
#     print("127 test data path ", test_data_path)

#     models = {
#             "Linear Regression": LinearRegression(),
#             "Lasso": Lasso(),
#             "Ridge": Ridge(),
#             "K-Neighbors Regressor": KNeighborsRegressor(),
#             "Decision Tree": DecisionTreeRegressor(),
#             "Random Forest Regressor": RandomForestRegressor(),
#             "XGBRegressor": XGBRegressor(), 
#             # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
#             "AdaBoost Regressor": AdaBoostRegressor()
#             }
#     model_list = []
#     r2_list =[]
#     training_result, test_result = data_training(train_data_path, test_data_path, models)
#     model_train_mae, model_train_rmse, model_train_r2 = training_result
#     model_test_mae, model_test_rmse, model_test_r2 = test_result
    
#     for i in range(len(list(models))):
#         print(list(models.keys())[i])
#         model_list.append(list(models.keys())[i])
        
#         print('Model performance for Training set')
#         print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
#         print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
#         print("- R2 Score: {:.4f}".format(model_train_r2))

#         print('----------------------------------')
        
#         print('Model performance for Test set')
#         print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
#         print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
#         print("- R2 Score: {:.4f}".format(model_test_r2))
#         r2_list.append(model_test_r2)
        
#         print('='*35)
#         print('\n')
    
import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class DataTransformationConfig:
    #path to save preprocessor.pkl file
    preprocessor_obj_file_path =  os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        #object of class DataTransformationConfig
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        this method is responsible for data transormation based on the different types of data.
        '''
        try:
            #segregating numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            
            #adding missing values using Simple imputer and normalizing data using StandardScaler
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            #Simple Imputer add missing values in data
            #one hot encoder encode the categorical data
            #Standard Scaler normalizes the data
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns standar scaling completed")
            logging.info("Categorical columns encoding completed")
            
            #applying above pipeline to column transfer model
            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def inititate_data_transformation(self,train_path,test_path):
        try:
            #reading training data
            train_df = pd.read_csv(train_path)
            #reading test dataset
            test_df = pd.read_csv(test_path)

            logging.info("read train and test data completed")

            logging.info("obtaining processing oject")
            
            #creating preprocessing object to implement preprocessing on training and test data
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]
            
            #splitting training data into predictors and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            targte_feature_train_df = train_df[target_column_name]
            
            #splitting test data into predictors and target
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            targte_feature_test_df = test_df[target_column_name]

            logging.info("applying preprocessing object on training dataset and testing dataset")
            
            #implementing preprocessing steps on training and test predictors
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            #Concatinating preprocessed predictors with the target
            train_arr = np.c_[input_feature_train_arr,np.array(targte_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(targte_feature_test_df)]

            logging.info("saved processing object")
            
            #saving the preprocessing data into preprocessor.pkl file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
                )
            
            #returning the concatenated arrays and preprocessor file path
            return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path, 
            )

        except Exception as e:
            raise CustomException(e,sys)







    
    
    
    