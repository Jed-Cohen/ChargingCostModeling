import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def regress(filename):
    df = pd.read_csv(filename)
    y = df['Converted Cost'].to_numpy()
    del df['Converted Cost']
    x = df.to_numpy()
    model = LinearRegression().fit(x, y)
    r_squared = model.score(x, y)
    print(f"coefficient of determination: {r_squared}")
    print(f"intercept: {model.intercept_}")
    print(f"coefficients: {model.coef_}")


def support_vector(filename):
    df = pd.read_csv(filename)
    y = df['Converted Cost']
    del df['Converted Cost']
    svr = svm.SVR()
    parameters = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10]
    }
    cv = GridSearchCV(svr, parameters, cv=5)
    cv.fit(df, y.values.ravel())
    print_results(cv)


def Stochastic_Gradient_Descent(filename):
    x = pd.read_csv(filename)
    y = x['Converted Cost']
    del x['Converted Cost']
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
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
    x = pd.read_csv(filename)
    y = x['Converted Cost']
    del x['Converted Cost']
    rf = RandomForestRegressor()
    parameters = {
        'n_estimators': [5, 50, 250],
        'max_depth': [2, 4, 8, 16, 32, None]
    }
    cv = GridSearchCV(rf, parameters, cv=5)
    cv.fit(x, y.values.ravel())
    print_results(cv)



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
    Stochastic_Gradient_Descent('data/level2final.csv')
    # Random_Forest('data/level2final.csv')