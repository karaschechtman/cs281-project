import numpy as np
import pandas as pd
import os 
import ot

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DATASET_FILENAME = os.getcwd() + '/../data/compas.csv'
PREDICTIONS_FILENAME = os.getcwd() + '/../data/compas_predictions.csv'
TEST_SIZE = 0.33
SEED = 123

def predict_unconstrained(X_train,y_train,X_test):
    estimator = LogisticRegression(solver="liblinear")
    estimator.fit(X_train, y_train)
    return [s[1] for s in estimator.predict_proba(X_test)]

def predict_calibrated(X_train,y_train,X_test):
    # TODO(kara): break apart by sensitive features
    estimator = LogisticRegression(solver="liblinear")
    calibrated_estimator = CalibratedClassifierCV(estimator,method='sigmoid',cv=2)
    calibrated_estimator.fit(X_train,y_train)
    return [s[1] for s in calibrated_estimator.predict_proba(X_test)]

def predict_EO_thresholdless(X_train,y_train,X_test,y_test,sensitive_features):
    df = X_test.copy()
    df['unconstrained'] = predict_unconstrained(X_train,y_train,X_test)
    scores = {}
    for y in y_train.unique():
        # Find the optimal transport plan between test distributions
        c = np.array(np.transpose(df[(sensitive_features=='Caucasian') & (y_test==y)]['unconstrained'])).reshape(-1,1)
        b = np.array(np.transpose(df[(sensitive_features=='African-American') & (y_test == y)]['unconstrained'])).reshape(-1,1)
        costs = ot.dist(c,b)
        c_u, b_u = np.ones(len(c)) / len(c), np.ones(len(b)) / len(b)  # uniform distribution on samples
        opt_transport = ot.emd(c_u, b_u, costs)

        # Assemble the new scores using the optimal transport matrix
        c_new, b_new = np.zeros(len(c)),np.zeros(len(b))
        for c_i in range(len(opt_transport)):
            for b_i in np.nonzero(opt_transport[c_i])[0]:
                c_score = c[c_i][0]
                b_score = b[b_i][0]
                mean_score = (c_score + b_score)/2
                c_new[c_i] = mean_score
                b_new[b_i] = mean_score
        scores['c_%d' % y] = c_new
        scores['b_%d' % y] = b_new

    # Save new scores
    for y in y_test.unique():
        df.loc[(sensitive_features=='Caucasian') & (y_test==y), 'thresholdless_EO'] = scores['c_%d' % y]
        df.loc[(sensitive_features=='African-American') & (y_test==y), 'thresholdless_EO'] = scores['b_%d' % y]
    
    return df['thresholdless_EO']

def _EO_class(row,label):
    if row['two_year_recid']==label:
        return row['thresholdless_EO']
    return row['unconstrained']

if __name__ == '__main__':
    dataset = pd.read_csv(DATASET_FILENAME)
    targets = dataset['two_year_recid']
    dataset = dataset.drop('two_year_recid',axis=1)
    X_train, X_test, y_train, y_test = train_test_split(dataset, targets, 
                                                       test_size=TEST_SIZE, 
                                                       random_state=SEED)
    race_train = X_train['race']
    race_test = X_test['race']
    X_train = pd.get_dummies(X_train.drop(['race','sex'],axis=1))
    X_test = pd.get_dummies(X_test.drop(['race','sex'],axis=1))
    df = X_test.copy()
    df['unconstrained'] = predict_unconstrained(X_train,y_train,X_test)
    df['calibrated'] = predict_calibrated(X_train,y_train,X_test)
    df['thresholdless_EO'] = predict_EO_thresholdless(X_train,y_train,X_test,
                                                      y_test,race_test)
    df['two_year_recid'] = y_test
    df['thresholdless_EO_pos_class'] = df.apply(_EO_class,args=[1],axis=1)
    df['thresholdless_EO_neg_class'] = df.apply(_EO_class,args=[0],axis=1)
    df['race'] = race_test
    df.to_csv(PREDICTIONS_FILENAME,index=False)