
from data_ingestion.data_loader import *
from model_training.models import ClassificationModels


data_path = input("Enter the path of the data: ")
data_loader = DataLoader(data_path)
data = data_loader.load_data()
target = input("Enter the target column: ")
data_split = DataSplit(data, target)
X_train, X_test, y_train, y_test = data_split.data_split()

print("Data Shape: ", data.shape)
print("X_train: ", X_train.shape)   
print("X_test: ", X_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)

print("----------------------Model Training Started-----------------")

model = ClassificationModels(X_train, X_test, y_train, y_test)
log_model, acc_score = model.LogisticRegression()

print(acc_score)

print("----------------------Model Training Completed-----------------")

