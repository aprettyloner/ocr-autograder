from preprocess import preprocess_json
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,KFold,GridSearchCV 

def plot_ROC(ns_fpr, ns_tpr, title=''):
    ns_probs = [0 for _ in range(len(y_test))]
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    plt.xlabel('False Positive Rate',fontsize=25)
    plt.ylabel('True Positive Rate',fontsize=25)
    plt.title(title,fontsize=35)


def build_XGBClassifier(X,y,plot=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_clf = XGBClassifier(subsample=0.6,learning_rate=0.07, max_depth=5)
    xgb_clf.fit(X_train, y_train)
    if plot:
        score = xgb_clf.score(X_test, y_test)
        print(score)
        probs = xgb_clf.predict_proba(X_test)
        lr_probs = probs[:, 1]
        ns_auc = roc_auc_score(y_test, ns_probs)
        lr_auc = roc_auc_score(y_test, lr_probs)
        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('Logistic: ROC AUC=%.3f' % (lr_auc))
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
        plot_ROC(ns_fpr, ns_tpr,title='XGBoost Classifier')
    return xgb_clf

def build_GBClassifier(X,y,plot=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.75, max_depth=5, random_state=0)
    gb_clf.fit(X_train, y_train)
    if plot:
        probs = gb_clf.predict_proba(X_test)
        lr_probs = probs[:, 1]
        ns_auc = roc_auc_score(y_test, ns_probs)
        lr_auc = roc_auc_score(y_test, lr_probs)
        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('Logistic: ROC AUC=%.3f' % (lr_auc))
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
        plot_ROC(ns_fpr, ns_tpr,title='Gradient Boosted Classifier')
    return gb_clf
    

if __name__ == '__main__':
    train_data,raw_df = preprocess_json('data/data.json',train=True)
    X = np.array(train_data.drop('Fraud',axis=1))
    y = np.array(train_data['Fraud'])

    xgb_model = build_XGBClassifier(X,y)
    gb_model = build_GBClassifier(X,y)
    
    with open('static/XGB_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)    

    with open('static/GB_model.pkl', 'wb') as f:
        pickle.dump(gb_model, f)    
