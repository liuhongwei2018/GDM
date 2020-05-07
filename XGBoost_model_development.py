import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV, StratifiedKFold, KFold
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import roc_auc_score


data_all = pd.read_csv("G:/GDM/DATA/GDM.csv")
X ,y= data_all.drop(['OGTTgroup1'],axis=1),data_all.OGTTgroup1
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 1,test_size=0.3,stratify=y)

cv_params_list = [{'n_estimators': [200,300,400]},
                  {'max_depth': range(2, 10), 'min_child_weight': range(3,20,2)},
                  {'colsample_bytree': [x / 10 for x in range(1, 11)], 'subsample': [x / 10 for x in range(5, 11)]},
                  {'gamma': [x / 10 for x in range(0, 51, 2)]},
                  {'reg_alpha': [15, 9, 5, 1, 0.1, 0, 0.5], 'reg_lambda': [15, 9, 5, 2, 1, 0, 0.5]},
                  {'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]}
                  ]
scale_pos_weight = (len(y) - sum(y)) / sum(y)
cv_params_list_copy = cv_params_list.copy()
ind_params = {'scale_pos_weight': scale_pos_weight}
old_params = ind_params.copy()
EPOCH = 16
K = 5
StKFold = StratifiedKFold(n_splits=K, random_state=999, shuffle=True)
best_params_ = {}

for e in range(EPOCH):
    for cv_params in cv_params_list_copy:
        GS_XGB = GridSearchCV(XGBClassifier(**ind_params,n_jobs=-1,random_state = 2644),cv_params,scoring='roc_auc',cv=StKFold,n_jobs=4)
        GS_XGB.fit(X_train,y_train)
        for key,value in GS_XGB.best_params_.items():
            ind_params[key] = value
        print("try to find: {}\n  score:{}".format(cv_params.keys(),GS_XGB.best_score_))
    print("The best params found in epoch {} is:\n{}".format(e, ind_params))
    if (old_params == ind_params) and (e >= 3):
        best_params_['GDM'] = ind_params
        break
    old_params = ind_params.copy()
    random.shuffle(cv_params_list_copy)
print('----{}_{} TRAINING FINISHED----'.format('GDM','1'))
best_params_file = pd.DataFrame.from_dict(best_params_)


seed = 1
clf = XGBClassifier(random_state=5361,scale_pos_weight=12.026280323450134,n_estimators=200,max_depth=2,
                    min_child_weight=29,colsample_bytree=0.7,subsample=1,gamma=0,
                    reg_alpha=5,reg_lambda=5,learning_rate=0.1, n_jobs=-1).fit(X_train, y_train)
pred_train = clf.predict_proba(X_train)
pred_test = clf.predict_proba(X_test)
score_train = roc_auc_score(y_train,pred_train[:,1])
score_test = roc_auc_score(y_test, pred_test[:,1])
#pd.concat([y_test,pred_test[:,1]])
score = [seed,score_train,score_test]
print([score])

clf.save_model("G:/Thesis/code/xgboost.model")

