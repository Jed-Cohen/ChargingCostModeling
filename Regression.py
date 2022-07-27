import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
import sklearn.metrics as sm
import joblib


# can be copied and modified for other potential regression algorithms
def xgboost(filename):
    features = pd.read_csv(filename)
    labels = np.array(features['Converted Cost'])
    features = features.drop('Converted Cost', axis=1)
    # Drop features manually here for feature selection
    # features = features.drop('Sales Tax', axis=1)
    # features = features.drop('Station Count', axis=1)
    # features = features.drop('Land Value', axis=1)
    # features = features.drop('Electric Price', axis=1)
    # features = features.drop('Economic Activity', axis=1)
    features = features.drop('ZIP', axis=1)
    features = np.array(features)
    # modify k for automatic feature selection
    features = SelectKBest(f_regression, k='all').fit_transform(features, labels)
    feature_list = list(features.columns)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=42)
    # Hyperparameters are arbitrary, they should be tuned for all models
    model = XGBRegressor(max_depth=7, learning_rate=.4, min_child_weight=.5)
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    baseline_errors = abs(labels.mean() - test_labels)
    # model metrics
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
    # joblib.dump(model, "USA_model.sav")  # uncomment this to save a model


if __name__ == '__main__':
    # Runs xgboost for datafile
    xgboost('Data/USA_DCFC.csv')
