from sklearn.model_selection import permutation_test_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, permutation_test_score
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

data_all = pd.read_csv("G:/GDM/DATA/GDM.csv")

X ,y= data_all.drop(['OGTTgroup1'],axis=1),data_all.OGTTgroup1
X_log ,y_log= data_all.drop(['OGTTgroup1','weight_gain','income','education','DBP',
                             'parity','multi_pregnancy'],axis=1),data_all.OGTTgroup1

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 1,test_size=0.3,stratify=y)
X_train_log,X_test_log,y_train_log,y_test_log = train_test_split(X_log,y_log,random_state = 1,test_size=0.3,stratify=y_log)

clf = XGBClassifier(random_state=1,n_jobs=-1).fit(X_train, y_train)
cv = StratifiedKFold(5)

score, permutation_scores, pvalue = permutation_test_score(
    clf, X_train, y_train, scoring="roc_auc", cv=cv, n_permutations=1000, n_jobs=-1)

#print(pvalue)

clf_log = LogisticRegression(random_state=0,fit_intercept=True, C=1e9,solver = 'newton-cg').fit(X_train_log, y_train_log)

score_log, permutation_scores_log, pvalue_log = permutation_test_score(
    clf_log, X_train_log, y_train_log, scoring="roc_auc", cv=cv, n_permutations=1000, n_jobs=-1)
#print(pvalue_log)
