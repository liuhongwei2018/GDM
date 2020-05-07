from sklearn.model_selection import permutation_test_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, permutation_test_score
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

data_all = pd.read_csv("G:/GDM/DATA/GDM.csv")
X ,y= data_all.drop(['OGTTgroup1'],axis=1),data_all.OGTTgroup1
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 1,test_size=0.3,stratify=y)
print(X_train.shape,sum(y_train))
print(X_test.shape, sum(y_test))

clf = XGBClassifier(random_state=1,n_jobs=-1).fit(X_train, y_train)
cv = StratifiedKFold(5)

score, permutation_scores, pvalue = permutation_test_score(
    clf, X_train, y_train, scoring="roc_auc", cv=cv, n_permutations=2000, n_jobs=-1)

#print(pvalue)
