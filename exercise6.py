import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

C = 1.0
h = .02

iris = datasets.load_iris()

names = iris.feature_names
print("Dataset feature label : ", names)

lin_svc = svm.SVC (kernel='linear', C=C )
rbf_svc = svm.SVC (kernel='rbf', gamma=0.7, C=C)
poly_svc = svm.SVC (kernel='poly', degree=3, C=C)

X_train,X_test,y_train,y_test = train_test_split(iris.data, iris.target, test_size=0.3)

lin_svc.fit(X_train, y_train)
rbf_svc.fit(X_train, y_train)
poly_svc.fit(X_train, y_train)

print("\nX train : ",X_train)
print("\nX test : ",X_test)
print("\nY train : ", y_train)
print("\nY test : ",y_test)

lin_predict = lin_svc.predict(X_test)
print ("\nLinear :")
print ("\nY test actual : \n", y_test)
print ("\nY test predicted : \n", lin_predict)
print ("\n Linear accuracy : ", (accuracy_score(y_test,lin_predict, normalize=True))*100,"% ")

print ("\nRBF :")
rbf_predict = rbf_svc.predict(X_test)
print ("\nY test actual : \n", y_test)
print ("\nY test predicted \n: ", rbf_predict)
print ("\n RBF accuracy : ", (accuracy_score(y_test,rbf_predict, normalize=True))*100,"% ")

print ("\nPoly  :")
poly_predict = poly_svc.predict(X_test)
print ("\nY test actual : \n", y_test)
print ("\nY test predicted : \n", poly_predict)
print ("\n Poly accuracy : ", (accuracy_score(y_test,poly_predict, normalize=True))*100,"% ")
