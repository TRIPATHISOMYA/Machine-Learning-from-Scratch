import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

iris_data = pandas.read_csv('../data/iris_data.csv')

# Split-out validation dataset

data = iris_data.values
X = data[:,0:4]
Y = data[:,4]

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20, random_state=7)


models = []
models.append(('Regression', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Decision', DecisionTreeClassifier()))
models.append(('Naive', GaussianNB()))
models.append(('SVM', SVC()))

# Model Evaluation
results = []
mod_name = []
for name, model in models:
   kfold = model_selection.KFold(n_splits=10, random_state=7)
   cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
   results.append(cv_results)
   mod_name.append(name)
   res = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
   print(res)



# Prediction
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))






