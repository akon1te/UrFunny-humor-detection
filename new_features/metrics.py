from sklearn.metrics import accuracy_score, precision_score, recall_score

def metrics(y_pred, y_true):
    
    return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred)