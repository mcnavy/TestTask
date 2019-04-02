from PIL import Image
import numpy as np
import glob
import collections
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import svm


def getData(files):
    X = []
    y = []
    files = ['closedLeftEyes/*jpg', 'closedRightEyes/*jpg', 'openLeftEyes/*jpg', 'openRightEyes/*jpg']
    for f in files:
        for filename in glob.glob(f):
            im = Image.open(filename)

            pixels = np.asarray(im, dtype='uint8')

            new_p = []
            for a in pixels:
                for aa in a:
                    new_p.append(aa)

            X.append(new_p)
            if 'closed' not in f:
                y.append(1)
            else:
                y.append(0)
    return X, y


def getMetrics(y_test, y_pred_proba, predictions, methodName):
    counter = 0
    for i in range(len(predictions)):
        if predictions[i] == y_test[i]:
            counter += 1
    print("Accuracy for {} method:".format(methodName), "{}/{} = ".format(counter, len(y_test)), counter / len(y_test))
    CM = confusion_matrix(y_test, predictions)
    print(CM)
    print(collections.Counter(y_test))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    plt.text(0.6, 0, "AUC = " + str(auc))
    plt.show()


def KNNClassifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    neigh = KNeighborsClassifier(n_neighbors=3)

    neigh.fit(X_train, y_train)

    predictions = neigh.predict(X_test)
    y_pred_proba = neigh.predict_proba(X_test)[::, 1]
    return predictions, y_pred_proba, y_test
def SVMClassifier(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    clf = svm.SVC(gamma='scale',probability=True)
    clf.fit(X, y)
    pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[::, 1]
    return pred,y_pred_proba,y_test

files = ['closedLeftEyes/*jpg', 'closedRightEyes/*jpg', 'openLeftEyes/*jpg', 'opengit RightEyes/*jpg']
X, y = getData(files)
predictionSVM,y_pred_probaSVM,y_testSVM = SVMClassifier(X,y)
getMetrics(y_testSVM,y_pred_probaSVM,predictionSVM,'SVM')

predictionsKNN, y_pred_probaKNN, y_testKNN = KNNClassifier(X, y)
getMetrics(y_testKNN, y_pred_probaKNN, predictionsKNN, 'KNN')
