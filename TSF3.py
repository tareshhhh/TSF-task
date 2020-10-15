#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

#loading dataset

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df
#feature selection
feature_cols = ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
X = iris_df.iloc[:,[0,1,2,3]].values
y = iris.target

#spliting dataset into training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.25, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#decision tree classifier

from sklearn.tree import DecisionTreeClassifier
cls = DecisionTreeClassifier()
cls = cls.fit(X_train, y_train)

#prediction
y_pred = cls.predict(X_test)
y_pred

#finding accuracy of model
from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))

#cofusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred )
cm


# Import necessary libraries for graph viz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

# Visualize the graph
dot_data = StringIO()
export_graphviz(cls, out_file=dot_data, feature_names=iris.feature_names,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())