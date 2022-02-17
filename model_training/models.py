"""Model Training"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class ClassificationModels:
    """This class will responsible for classification model training.
    """
    
    def __init__(self, X_train,X_test,y_train,y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def LogisticRegression(self):
        log_reg =LogisticRegression()
        log_reg.fit(self.X_train,self.y_train)
        prediction = log_reg.predict(self.X_test)
        acc_score = accuracy_score(self.y_test,prediction)*100
    
        return log_reg, acc_score
             
        
    def DecisionTree(self):
        pass
    
    def RandomForest(self):
        pass
    
    def KNN(self):
        pass
    
    def XgBoost(self):
        pass
       
       
class RegressionModels:
    """This class will responsible for regression model training.
    
    """
    pass


       