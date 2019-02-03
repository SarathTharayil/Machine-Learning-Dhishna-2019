import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt

iris_flower = datasets.load_iris()

names = iris_flower.feature_names
print("Dataset feature label : ", names)
# print("Description : ", iris_flower['DESCR'])
# print("Feature Data : ", iris_flower['data'])

X = iris_flower.data[:,:2]
# print(X)

y = iris_flower.target
plt.scatter(X[:, 0],X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width" )
plt.title("Sepal Width and Length : ")
plt.show()

Z = iris_flower.data[: , 2:]
# print(X)

v = iris_flower.target
plt.scatter(Z[:, 0], Z[:, 1], c=v, cmap=plt.cm.coolwarm)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width" )
plt.title("Petal Width and Length")
plt.show()
