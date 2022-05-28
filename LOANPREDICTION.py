# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 19:15:09 2022

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics  import mean_absolute_error
loan = pd.read_csv('C:/Users/USER/Downloads/train_ctrUa4K.csv')
dum_loan=pd.get_dummies(loan,drop_first=True)
############to remove nan values#########3
loan = loan.dropna()
loan.dropna(inplace = True)
np.sum(pd.isnull(loan))

########seaborn
import seaborn as sns
sns.countplot('Loan_Status',data=loan)
plt.show()

sns.countplot('Education',data=loan)
plt.show()

sns.countplot( 'Gender',data=loan)
plt.show()

sns.countplot('Property_Area',data=loan)
plt.show()

# Heatmap
sns.heatmap( loan.corr(),annot=True)
plt.show()

sns.heatmap( loan.corr(),cmap="YlGnBu",annot=True)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()
############ Facets

g = sns.FacetGrid(loan, col="Loan_Status")
g.map(plt.hist, "Property_Area")
plt.show()

g = sns.FacetGrid(loan, row="ApplicantIncome")
g.map(plt.hist, "Loan_status")
plt.show()

g = sns.FacetGrid(loan, col="Dependents")
g.map(plt.scatter, "ApplicantIncome",
                   "Loan_Status")
plt.show()
############count plot

cts = loan['Self_Employed'].value_counts()
cts.plot(kind='bar', color="Blue")
plt.show()

cts = loan['Dependents'].value_counts()
cts.plot(kind='bar', color="Blue")
plt.show()

cts.plot(kind='pie')
plt.show() 

cts.plot(kind='pie',autopct="%.2f%%")
plt.show() 
 
cts.plot(kind='pie',autopct="%.2f%%",shadow=True)
plt.show() 
################Density plot
loan['ApplicantIncome'].plot(kind='kde')
plt.show()

loan['LoanAmount'].plot(kind='kde')
plt.show()

################3NAIVE BAYES
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


loan = pd.read_csv("C:/Users/USER/Downloads/train_ctrUa4K.csv")
dum_loan = pd.get_dummies(loan, drop_first=True)

d1 = loan.iloc[:,1:-2]
d2 = loan.iloc[:,-1]

dum_d1=pd.get_dummies(d1, drop_first=True)
dum_d2=pd.get_dummies(d2, drop_first=True)

X = dum_d1
y = dum_d2
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=2022,
                                                    stratify=y)

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)

############## Model Evaluation ##############
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(gaussian, X_test, y_test)  # doctest: +SKIP
plt.show() 

# ROC
from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_probs = gaussian.predict_proba(X_test)
y_pred_prob = y_probs[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

roc_auc_score(y_test, y_pred_prob)
##############LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.coef_)
print(regressor.intercept_)

y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=5, random_state=2022,shuffle=True)

regressor = LinearRegression()

results = cross_val_score(regressor, X, y, cv=kfold)
print(results)
print("R2: %.4f (%.4f)" % (results.mean(), results.std()))

results = cross_val_score(regressor, X, y, cv=kfold,
                          scoring='neg_mean_squared_error')
print(results)
print("Neg MSE: %.4f (%.4f)" % ((-1)*results.mean(), results.std()))
##############DECISION TREE FOR CLASSIFICATION
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=2022,max_depth=3)

#clf = DecisionTreeClassifier(max_depth=3,random_state=2018,
#                           min_samples_split=20,min_samples_leaf=5)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))
################ROC#############################

# Import necessary modules
from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = clf.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
roc_auc_score(y_test, y_pred_prob)

################################################################
import graphviz 


from sklearn import tree
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=X_train.columns,  
                         class_names=['No','Yes'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

#######################Grid Search CV###########################
depth_range = [3,4,5,6,7,8,9]
minsplit_range = [5,10,20,25,30]
minleaf_range = [5,10,15]

parameters = dict(max_depth=depth_range,
                  min_samples_split=minsplit_range, 
                  min_samples_leaf=minleaf_range)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
clf = DecisionTreeClassifier(random_state=2022)

kfold = StratifiedKFold(n_splits=5, random_state=2022,shuffle=True)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='roc_auc')

cv.fit(X,y)
# Best Parameters
print(cv.best_params_)

print(cv.best_score_)

best_model = cv.best_estimator_
from sklearn import tree
dot_data = tree.export_graphviz(best_model, out_file=None, 
                         feature_names=X_train.columns,  
                         class_names=['No','Yes'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 
########################################################
import matplotlib.pyplot as plt

best_model.feature_importances_

ind = np.arange(X.shape[1])
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,(X.columns))
plt.title('Feature Importance')
plt.xlabel("Variables")
plt.show()
#######################################################

pd.crosstab(index=loan['loan'],
            columns=cv['ApplicantIncome'],
            margins=True)

loan.groupby('loan')["ApplicantIncome"].mean()
