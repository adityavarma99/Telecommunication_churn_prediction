import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")  # Save preprocessor object


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns):
        """
        This function creates a preprocessing pipeline for numerical features.
        """
        try:
            # Define numerical preprocessing pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Handle missing values
                    ("scaler", StandardScaler())  # Scale numerical features
                ]
            )
            logging.info("Numerical pipeline created successfully.")

            # ColumnTransformer to handle numerical columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns)  # Apply pipeline to specified numerical columns
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Transforms the train and test datasets using the preprocessing pipeline.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Debugging step: Print columns
            print("Train columns:", train_df.columns)
            print("Test columns:", test_df.columns)

            logging.info("Read train and test data completely")

            target_column_name = "churn"
            numerical_columns = [
                'account_length', 'voice_mail_plan', 'voice_mail_messages', 'day_mins', 'evening_mins', 'night_mins',
                'international_mins', 'customer_service_calls', 'international_plan', 'day_calls', 'day_charge',
                'evening_calls', 'evening_charge', 'night_calls', 'night_charge', 'international_calls',
                'international_charge', 'total_charge'
            ]

            # Ensure the columns exist
            assert all(col in train_df.columns for col in numerical_columns), "Some numerical columns are missing!"

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(numerical_columns)

            # Splitting features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)  # X train
            target_feature_train_df = train_df[target_column_name]  # y train

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)  # X test
            target_feature_test_df = test_df[target_column_name]  # y test

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            # Transform features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features with target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            # Save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error(f"Error during data transformation: {e}")
            raise CustomException(e, sys)
