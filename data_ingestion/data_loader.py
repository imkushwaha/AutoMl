"""We have to take data path as a input from user and read it in data frame.
   Args: Path
   Return: DataFrame
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:

    def __init__(self, path):
        self.path = path

    def load_data(self):
        """
        Load data from path
        """
        data = pd.read_csv(self.path)
        return data

class DataSplit:

    def __init__(self,data, target):
        self.data = data
        self.target = target

    def data_split(self):
        """
        To split the data into train and test
        """
        X = self.data.drop(self.target, axis = 1)
        y = self.data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test



