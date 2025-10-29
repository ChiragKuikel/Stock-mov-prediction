from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
def evaluate_model(model,x_test,y_test):
    y_pred = model.predict(x_test)
    ar = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return ar,cr,cm
