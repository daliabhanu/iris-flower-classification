import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#load the iris dataset
iris=load_iris()
x=iris.data
y=iris.target
#split data into training and testsets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#featue scaling
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
#knn classifier (a simple and effective algorithm for classifier)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
#making predictions
y_pred=knn.predict(x_test)
#evaluate the model
accuracy=accuracy_score(y_test,y_pred)
print("Accuacy:",accuracy)
print("classification Report:")
print(classification_report(y_test,y_pred))
print("confusion matrix:")
print(confusion_matrix(y_test,y_pred))