import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
import sklearn.metrics as sm
import joblib


def regress(filename):
    df = pd.read_csv(filename)
    x = (df-df.mean())/df.std()
    y = x['Converted Cost']
    del x['Converted Cost']
    model = LinearRegression().fit(x, y)
    r_squared = model.score(x, y)
    print(f"coefficient of determination: {r_squared}")
    print(f"intercept: {model.intercept_}")
    print(f"coefficients: {model.coef_}")


def support_vector(filename):
    df = pd.read_csv(filename)
    x = (df - df.mean()) / df.std()
    y = x['Converted Cost']
    del x['Converted Cost']
    svr = svm.SVR()
    parameters = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10]
    }
    cv = GridSearchCV(svr, parameters, cv=5)
    cv.fit(x, y.values.ravel())
    print_results(cv)


def Stochastic_Gradient_Descent(filename):
    df = pd.read_csv(filename)
    x = (df-df.mean())/df.std()
    y = x['Converted Cost']
    del x['Converted Cost']
    parameters = {
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], # learning rate
        'max_iter': [1000], # number of epochs
        'penalty': ['l2'],
    }
    scg = SGDRegressor()
    cv = GridSearchCV(scg, parameters, cv=5)
    cv.fit(x, y.values.ravel())
    print_results(cv)


def Random_Forest(filename):
    features = pd.read_csv(filename)
    labels = np.array(features['Converted Cost'])
    features = features.drop('Converted Cost', axis=1)
    features = features.drop('Number of Evs', axis=1)
    features = features.drop('Population', axis=1)
    # features = features.drop('Sales Tax', axis=1)
    # features = features.drop('EV density', axis=1)
    # features = features.drop('Station Count', axis=1)
    features = features.drop('ZIP', axis=1)


    feature_list = list(features.columns)
    features = np.array(features)
    features = SelectKBest(f_regression, k='all').fit_transform(features, labels)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=42)
    # baseline_preds = test_features[:, feature_list.index('Average')]
    # baseline_errors = abs(baseline_preds - test_labels)
    rf = RandomForestRegressor(n_estimators=250, max_depth=4, random_state=42)
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)
    print('Average baseline error: ', round(np.mean(abs(labels.mean() - test_labels)), 2))
    print("Mean absolute error =", round(sm.mean_absolute_error(test_labels, predictions), 2))
    print("Mean squared error =", round(sm.mean_squared_error(test_labels, predictions), 2))
    print("Median absolute error =", round(sm.median_absolute_error(test_labels, predictions), 2))
    print("Explain variance score =", round(sm.explained_variance_score(test_labels, predictions), 2))
    print("R2 score =", round(sm.r2_score(test_labels, predictions), 2))
    importances = list(rf.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

def xgboost(filename):
    features = pd.read_csv(filename)
    labels = np.array(features['Converted Cost'])
    features = features.drop('Converted Cost', axis=1)
    # features = features.drop('Number of Evs', axis=1)
    # features = features.drop('Population', axis=1)
    # features = features.drop('Sales Tax', axis=1)
    features = features.drop('Density', axis=1)
    # features = features.drop('Station Count', axis=1)
    # features = features.drop('Land Value', axis=1)
    features = features.drop('Electric Price', axis=1)
    # features = features.drop('Economic Activity', axis=1)
    features = features.drop('ZIP', axis=1)
    feature_list = list(features.columns)
    features = np.array(features)
    features = SelectKBest(f_regression, k='all').fit_transform(features, labels)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=42)
    model = XGBRegressor(max_depth=7, learning_rate=.3, min_child_weight=.5)
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    baseline_errors = abs(labels.mean() - test_labels)
    print('Average baseline error: ', round(np.mean(baseline_errors), 2))
    print("Mean absolute error =", round(sm.mean_absolute_error(test_labels, predictions), 2))
    print("Mean squared error =", round(sm.mean_squared_error(test_labels, predictions), 2))
    print("Median absolute error =", round(sm.median_absolute_error(test_labels, predictions), 2))
    print("Explain variance score =", round(sm.explained_variance_score(test_labels, predictions), 2))
    print("R2 score =", round(sm.r2_score(test_labels, predictions), 2))
    importances = list(model.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    # joblib.dump(model, "USA_model.sav")


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))


if __name__ == '__main__':
    # regress('data/level2final.csv')
    # regress('data/DCFCfinal.csv')
    # support_vector('Data/level2final.csv') # doesn't work
    # Stochastic_Gradient_Descent('data/level2final.csv')
    # Random_Forest('Data/CA advanced.csv')
    xgboost('Data/DCFC density data.csv')
