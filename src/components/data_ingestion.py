import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig():
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    
class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("started reading dataset")
        
        try:
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("dataset read successfully")
            
            #creates artifacts folder
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            #saving dataset to raw data path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("train test split started")
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
            
            #saving train data to train data path
            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            #saving test data to test data path
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data ingestion completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,tes_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.inititate_data_transformation(train_data,tes_data)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))

        