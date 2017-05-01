from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def extract_features(data, indices):
    return data[:, indices]


def calculate_model_accuracy(predict_train, predict_test, target_train, target_test):
    train_accuracy = metrics.accuracy_score(target_train, predict_train)
    test_accuracy = metrics.accuracy_score(target_test, predict_test)
    return (train_accuracy, test_accuracy)

def calculate_confusion_matrix(predict_test,target_test):
    return metrics.confusion_matrix(target_test,predict_test)

def create_decision_tree(max_features=None):
    '''Returns a sklearn.DecisionTreeClassifier with the max_features provided'''
    model = DecisionTreeClassifier(max_features=max_features)
    return model


def create_random_forest(n_estimators=10):
    '''Returns a sklearn.RandomForestClassifier with the n_estimators provided'''
    model = RandomForestClassifier(n_estimators=n_estimators)
    return model
