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


X = iris.data[:, :2]
y=iris.target
x_min,x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min,y_max = X[:, 1].min()-1, X[:, 1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))

titles = ["SVC with Linear Kernel", 'SVC with RBF Kernel', 'SVC with Polynomial (Degree 3) Kernel']

for i, clf in enumerate((lin_svc,rbf_svc,poly_svc)):
        plt.subplot(2, 2, i+1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshap(xx, shape)
        plt.contour(xx, yy,  Z, cmap=plt.cm.coolwarm, alpha=0.8)

        plt.scatter(X[: , 0], X[:, 1], c=y ,cmap=plt.cm.coolwarm)
        plt.xlabel("Sepal Length")
        plt.ylabel("Sepal Width" )
        plt.xlim(xx.min(),xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])
        plt.show()