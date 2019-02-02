import numpy as np 
from sklearn import datasets
iris_flower = datasets.load_iris()

# print(iris_flower)

feat_shape = iris_flower.data.shape
print("Dataset Size : ",feat_shape)
# Dataset size

names = iris_flower.feature_names
print("Dataset feature label : ",names)
# Dataset features

dataset = iris_flower.data
print("Actual dataset : \n",dataset)

print (iris_flower.target_names)

print (iris_flower.target)

print(iris_flower.target.shape)

