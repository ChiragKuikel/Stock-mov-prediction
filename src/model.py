from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier
import numpy as np
import joblib
def train_model_Xgboost(x_train,y_train):
    bst = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, objective='multi:softmax',num_class=3)
    bst.fit(x_train, y_train)
    return bst

def save_model(model,path):
    joblib.dump(model,path)
def load_model(path):
    return joblib.load(path)